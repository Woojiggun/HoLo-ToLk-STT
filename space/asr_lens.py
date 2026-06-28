"""asr_lens.py - char-CTC ASR for the HSL LENS experiment (HoLo-ToLk-STT).

What HSL ACTUALLY is: NOT "everything is bytes".
HSL UNFOLDS each byte into its 8 bits, then encodes the CHANGE-RATE (derivative) of
that bit pattern:
  dxor  = 1st-order change-rate   (Gray code  v ^ (v>>1)  = adjacent-bit difference)
  d2xor = 2nd-order change-rate   (delta of delta across consecutive bytes)
  + boundary, bit-FFT, phase.
=> HSL is a multi-order DERIVATIVE representation of bit-unfolded data, not raw bytes.

Why audio still needs a LENS: derivatives emphasize CHANGE (high-freq) but do NOT give
a frequency DECOMPOSITION the way STFT/mel does. Empirically HSL(audio) alone plateaus
~0.67-0.70 (vowel soup); mel ~0.21. Adding raw TIME-domain channels (hslwav) stayed ~0.70 --
the missing ingredient is specifically the time->frequency transform, not "more features".

Frontends (money chart -- identical char-CTC / Pre-LN / SortaGrad; differ ONLY in front-end):
  hsl     : in_norm(HSL(audio)) -> ConvSub                 [floor   ~0.67]  derivative substrate only
  mel     : log-mel -> proj                                [ceiling ~0.21]  spectral only
  hslwav  : [HSL(audio), wav, delta, |wav|] -> ConvSub     [        ~0.70]  + TIME-domain extras (inert)
  hslspec : ConvSub(HSL(audio))  +  mel_proj(log-mel)      [  THE TEST   ]  derivative substrate + SPECTRAL lens

hslspec is design (b): keep the LOSSLESS derivative substrate of the AUDIO bytes, and ADD
the spectral lens as a SEPARATE channel (NOT HSL-of-mel). It is a 1:1 contrast with hslwav --
same substrate, the only change is swapping the time-domain extras for the frequency lens --
so "frequency is the active ingredient" is shown with the substrate held constant, while the
lossless byte core is preserved (the unification story stays intact).

The spectral lens here is the model-side, parameter-free spectral feature lens described in
holo_core.hsl_signal (log-mel + a learnable fusion gate over the lossless HSL substrate); it
is vendored inline below (torchaudio MelSpectrogram + a learnable scalar gate) so this script
runs from `pip install hsl-embedding-zero` alone, with no extra package needed.

Data: *.jsonl.zst rows built by collect_libri.py: {"text","ulaw_b64","sr":8000}.
Import contract: hsl_embedding_zero only (frozen 0-param encoder); never import hsl_embedding.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import torch
import torch.nn as nn
import zstandard as zstd

import hsl_embedding_zero as _hez

hsl = _hez.hsl

# Default data directory is relative to this repo: ./data/librispeech100
# Override with --data, or set HOLO_DATA to point elsewhere.
DEFAULT_DATA = os.environ.get(
    "HOLO_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "librispeech100"),
)
SILENCE_U8 = 128                      # mu-law midpoint = silence; pad with this (NOT 0 = loud)
BLANK = 0
CHARS = " abcdefghijklmnopqrstuvwxyz'"
ID2CH = ["<blank>"] + list(CHARS)
CH2ID = {ch: i + 1 for i, ch in enumerate(CHARS)}
CONVS = [(10, 5, 3), (8, 4, 2), (4, 4, 1), (4, 2, 1)]   # stride 5*4*4*2 = 160 -> 8 kHz to ~50 fps


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_text(text: str) -> list[int]:
    return [CH2ID[ch] for ch in normalize_text(text) if ch in CH2ID]


def decode_ids(ids: list[int]) -> str:
    return "".join(ID2CH[i] for i in ids if 0 < i < len(ID2CH)).strip()


def load_speech(path: str) -> list[tuple[str, bytes, list[int]]]:
    if not os.path.exists(path):
        return []
    rows: list[tuple[str, bytes, list[int]]] = []
    with open(path, "rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as r:
            for line in io.TextIOWrapper(r, encoding="utf-8", errors="replace"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                text = normalize_text(rec.get("text", ""))
                labels = encode_text(text)
                ulaw_b64 = rec.get("ulaw_b64")
                if text and labels and ulaw_b64:
                    rows.append((text, base64.b64decode(ulaw_b64), labels))
    return rows


def conv_out_len(n: int) -> int:
    for k, s, p in CONVS:
        n = (n + 2 * p - k) // s + 1
    return max(1, n)


def mu_law_decode(ids: torch.Tensor) -> torch.Tensor:
    """u-law byte ids [0,255] -> waveform in [-1,1]."""
    mu = 255.0
    y = ids.float() / mu * 2.0 - 1.0
    return torch.sign(y) * (1.0 / mu) * ((1.0 + mu) ** y.abs() - 1.0)


class ConvSub(nn.Module):
    """1D conv subsampler, stride 160 (8 kHz -> ~50 fps)."""
    def __init__(self, c_in: int, dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        c = c_in
        for k, s, p in CONVS:
            layers += [nn.Conv1d(c, dim, k, s, p), nn.GELU()]
            c = dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, c_in, N] -> [B, dim, T]
        return self.net(x)


def sinusoidal_pe(T: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(T, device=device).float()[:, None]
    i = torch.arange(0, dim, 2, device=device).float()[None, :]
    ang = pos / (10000.0 ** (i / dim))
    pe = torch.zeros(T, dim, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class CharASR(nn.Module):
    def __init__(self, dim=384, layers=8, heads=6, ff=1536, frontend="hslspec", sr=8000, dropout=0.1, fuse="gate"):
        super().__init__()
        self.frontend = frontend
        self.dim = dim
        self.sr = sr
        self.specaug = False                                  # toggled from run() by --specaug
        self.use_sub = frontend in ("hsl", "hslwav", "hslspec")   # HSL(audio) derivative substrate path
        self.use_mel = frontend in ("mel", "hslspec")             # spectral lens path
        if not (self.use_sub or self.use_mel):
            raise ValueError(f"unknown frontend: {frontend}")

        if self.use_sub:
            # frozen 0-param encoder; momentum_phase=True puts |delta-byte| (velocity) on the phasor (signal path)
            self.hsl = hsl.Embedding(momentum_phase=True)
            self.in_norm = nn.LayerNorm(27)                   # per-feature norm at the model boundary (encoder mandates it)
            self.sub = ConvSub(30 if frontend == "hslwav" else 27, dim)
        if self.use_mel:
            import torchaudio
            self.hop = int(0.020 * sr)
            self.n_mels = 80
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr, n_fft=512, win_length=int(0.025 * sr),
                hop_length=self.hop, n_mels=self.n_mels,
            )
            self.mel_norm = nn.LayerNorm(self.n_mels)
            self.mel_proj = nn.Linear(self.n_mels, dim)

        self.fuse = fuse
        if self.use_sub and self.use_mel and fuse == "gate":          # hslspec gated fusion
            self.fuse_gate = nn.Parameter(torch.tensor(0.1))          # init small: mel dominates, substrate added iff it helps

        enc_layer = nn.TransformerEncoderLayer(
            dim, heads, ff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True,  # Pre-LN
        )
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(dim, len(ID2CH))

    def _time_mask(self, ids: torch.Tensor, n_bytes: list[int], n_masks=2, max_frac=0.10):
        for i, n in enumerate(n_bytes):
            if n < 4000:
                continue
            for _ in range(n_masks):
                w = random.randint(1, max(1, int(n * max_frac)))
                s = random.randint(0, max(0, n - w))
                ids[i, s:s + w] = SILENCE_U8
        return ids

    def _front(self, ids: torch.Tensor, n_bytes: list[int]):
        if self.specaug and self.training:
            ids = self._time_mask(ids.clone(), n_bytes)       # mask the shared waveform -> both paths see it

        h = None
        in_len = None
        Tsub = None
        if self.use_sub:
            feat = self.in_norm(self.hsl(ids))                # [B, N, 27]  derivative substrate (lossless)
            if self.frontend == "hslwav":
                wav = mu_law_decode(ids).unsqueeze(-1)
                delta = torch.zeros_like(wav)
                delta[:, 1:] = wav[:, 1:] - wav[:, :-1]
                feat = torch.cat([feat, wav, delta, wav.abs()], dim=-1)   # +3 time-domain channels (inert, for ablation)
            h = self.sub(feat.transpose(1, 2)).transpose(1, 2)            # [B, Tsub, dim]
            Tsub = h.shape[1]
            in_len = torch.tensor([conv_out_len(n) for n in n_bytes], device=ids.device)

        if self.use_mel:
            wav = mu_law_decode(ids)
            m = torch.log(self.mel(wav) + 1e-6).transpose(1, 2)          # [B, Tmel, 80]  spectral lens
            h_mel = self.mel_proj(self.mel_norm(m))                      # [B, Tmel, dim]
            if self.use_sub:                                            # hslspec: lens + (gated) substrate, aligned ~50 fps
                T = min(Tsub, h_mel.shape[1])
                sub = self.fuse_gate * h[:, :T] if self.fuse == "gate" else h[:, :T]
                h = h_mel[:, :T] + sub                                  # mel dominates; gate (init 0.1) adds substrate iff useful
                in_len = in_len.clamp(max=T)
            else:                                                       # mel only
                h = h_mel
                in_len = torch.tensor([n // self.hop + 1 for n in n_bytes], device=ids.device)

        return h, in_len.clamp(max=h.shape[1])

    def forward(self, ids: torch.Tensor, n_bytes: list[int]):
        h, in_len = self._front(ids, n_bytes)
        T = h.shape[1]
        h = h + sinusoidal_pe(T, self.dim, h.device)[None]
        pad_mask = torch.arange(T, device=h.device)[None, :] >= in_len[:, None]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.head(h).log_softmax(-1), in_len


def collate(batch, device: str):
    n_bytes = [len(ulaw) for _, ulaw, _ in batch]
    Nmax = max(n_bytes)
    ids = torch.full((len(batch), Nmax), SILENCE_U8, dtype=torch.long)   # pad with silence, not 0
    targets, target_lens = [], []
    for i, (_, ulaw, labels) in enumerate(batch):
        x = torch.tensor(list(ulaw), dtype=torch.long)
        ids[i, :len(x)] = x
        targets.extend(labels)
        target_lens.append(len(labels))
    return (ids.to(device), n_bytes,
            torch.tensor(targets, dtype=torch.long, device=device),
            torch.tensor(target_lens, dtype=torch.long, device=device))


def greedy_decode(logp: torch.Tensor, in_len: torch.Tensor) -> list[str]:
    pred = logp.argmax(-1)
    outs = []
    for b in range(pred.shape[0]):
        prev, ids = BLANK, []
        for t in pred[b, :int(in_len[b])].tolist():
            if t != prev and t != BLANK:
                ids.append(t)
            prev = t
        outs.append(decode_ids(ids))
    return outs


def edit_distance(a, b) -> int:
    da, db = len(a), len(b)
    dp = list(range(db + 1))
    for i in range(1, da + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, db + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[db]


@torch.no_grad()
def evaluate(model, data, device, batch_size, cap=0):
    model.eval()
    rows = data if cap <= 0 else data[:cap]
    ce = ct = we = wt = 0
    samples = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        X, nb, _t, _tl = collate(chunk, device)
        logp, in_len = model(X, nb)
        for (ref, _u, _l), hyp in zip(chunk, greedy_decode(logp, in_len)):
            ce += edit_distance(hyp, ref); ct += max(len(ref), 1)
            we += edit_distance(hyp.split(), ref.split()); wt += max(len(ref.split()), 1)
            if len(samples) < 5:
                samples.append((ref, hyp))
    return {"cer": ce / max(ct, 1), "wer": we / max(wt, 1), "n_eval": len(rows), "samples": samples}


@dataclass
class TrainConfig:
    data: str; train_file: str; heldout_file: str; frontend: str
    dim: int; layers: int; heads: int; ff: int
    batch: int; eval_batch: int; steps: int; lr: float; warmup: int
    eval_every: int; eval_cap: int; lr_floor: float
    max_bytes: int; limit_train: int; specaug: int; sortagrad: int; sr: int; seed: int; fuse: str


def save_checkpoint(path, model, opt, cfg, step, best, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "config": asdict(cfg),
        "vocab": {"blank": BLANK, "chars": CHARS, "id2ch": ID2CH},
        "step": step, "best_cer": best, "metrics": metrics,
        "hsl_embedding_zero": getattr(_hez, "__version__", "?"),
    }, path)


def load_model_from_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    model = CharASR(c["dim"], c["layers"], c["heads"], c["ff"], c["frontend"], c["sr"], fuse=c.get("fuse", "gate")).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def run(args):
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    data_dir = args.data or DEFAULT_DATA
    train_path = os.path.join(data_dir, args.train_file)
    held_path = os.path.join(data_dir, args.heldout_file)
    train = load_speech(train_path)
    heldout = load_speech(held_path)
    if not train or not heldout:
        raise SystemExit(f"missing data: {train_path} / {held_path}")

    if args.eval_only:
        model, _ck = load_model_from_checkpoint(args.eval_only, device)
        m = evaluate(model, heldout, device, args.eval_batch, args.eval_cap)
        print(f"[eval-only] {args.eval_only}\nCER {m['cer']:.4f} | WER {m['wer']:.4f} | n={m['n_eval']}")
        for ref, hyp in m["samples"]:
            print(f"REF: {ref[:100]}\nHYP: {hyp[:100]!r}")
        return

    if args.max_bytes:
        before = len(train)
        train = [r for r in train if len(r[1]) <= args.max_bytes]
        print(f"[filter] max_bytes={args.max_bytes}: train {before} -> {len(train)}")
    if args.limit_train:
        train = train[:args.limit_train]
        print(f"[filter] limit_train={args.limit_train}: train -> {len(train)}")

    cfg = TrainConfig(
        data=data_dir, train_file=args.train_file, heldout_file=args.heldout_file, frontend=args.frontend,
        dim=args.dim, layers=args.layers, heads=args.heads, ff=args.ff,
        batch=args.batch, eval_batch=args.eval_batch, steps=args.steps, lr=args.lr, warmup=args.warmup,
        eval_every=args.eval_every, eval_cap=args.eval_cap, lr_floor=args.lr_floor,
        max_bytes=args.max_bytes, limit_train=args.limit_train, specaug=args.specaug,
        sortagrad=args.sortagrad, sr=args.sr, seed=args.seed, fuse=args.fuse,
    )
    model = CharASR(args.dim, args.layers, args.heads, args.ff, args.frontend, args.sr, fuse=args.fuse).to(device)
    model.specaug = bool(args.specaug)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    best_path = args.ckpt or os.path.join(data_dir, f"asr_lens_best_{args.frontend}.pt")
    resume_path = args.resume_path or os.path.join(data_dir, f"asr_lens_resume_{args.frontend}.pt")
    best = float("inf")
    start_step = 0
    if args.resume and os.path.exists(resume_path):
        try:
            ck = torch.load(resume_path, map_location=device)
            model.load_state_dict(ck["model"])
            if ck.get("opt"):
                opt.load_state_dict(ck["opt"])
            start_step = int(ck.get("step", 0))
            best = float(ck.get("best_cer", best))
            print(f"[resume] {resume_path} step={start_step} best_cer={best:.4f}")
        except Exception as e:
            print(f"[resume] failed: {e}. starting fresh.")

    print(f"[asr_lens] device={device} front={args.frontend} train={len(train)} heldout={len(heldout)} "
          f"dim{args.dim}/L{args.layers}/h{args.heads}/ff{args.ff} params={nparam:.2f}M "
          f"steps={args.steps} bs={args.batch} eval_cap={'full' if args.eval_cap <= 0 else args.eval_cap}")

    rng = random.Random(args.seed)
    train_sorted = sorted(train, key=lambda r: len(r[1]))           # SortaGrad: short clips first
    sg_steps = (len(train) // args.batch) if args.sortagrad else 0
    t0 = time.time()
    for step in range(start_step, args.steps):
        model.train()
        if step < args.warmup:
            frac = (step + 1) / max(1, args.warmup)
        else:
            raw = 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
            frac = args.lr_floor + (1.0 - args.lr_floor) * raw
        for g in opt.param_groups:
            g["lr"] = args.lr * frac

        if step < sg_steps:
            s0 = (step * args.batch) % len(train_sorted)
            batch = train_sorted[s0:s0 + args.batch]
            if len(batch) < args.batch:
                batch += train_sorted[:args.batch - len(batch)]
        else:
            batch = [train[rng.randrange(len(train))] for _ in range(args.batch)]
        X, nb, tgt, tl = collate(batch, device)
        logp, in_len = model(X, nb)
        loss = ctc(logp.transpose(0, 1), tgt, in_len, tl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"step {step:6d} | loss {loss.item():.3f} | lr {args.lr * frac:.2e} | {(time.time()-t0)/60:.1f}m", flush=True)

        if args.eval_every and step and step % args.eval_every == 0:
            m = evaluate(model, heldout, device, args.eval_batch, args.eval_cap)
            better = m["cer"] < best
            if better:
                best = m["cer"]
                save_checkpoint(best_path, model, opt, cfg, step + 1, best, m)
            save_checkpoint(resume_path, model, opt, cfg, step + 1, best, m)
            gate = f" gate={model.fuse_gate.item():+.3f}" if hasattr(model, "fuse_gate") else ""
            print(f"[eval {step}] CER {m['cer']:.4f} | WER {m['wer']:.4f} | n={m['n_eval']}{gate} {'BEST' if better else ''}", flush=True)
            if m["samples"]:
                ref, hyp = m["samples"][0]
                print(f"REF: {ref[:100]}\nHYP: {hyp[:100]!r}", flush=True)

    m = evaluate(model, heldout, device, args.eval_batch, args.eval_cap)
    if m["cer"] < best:
        best = m["cer"]
        save_checkpoint(best_path, model, opt, cfg, args.steps, best, m)
    save_checkpoint(resume_path, model, opt, cfg, args.steps, best, m)
    print("\n=== asr_lens result ===")
    print(f"front={args.frontend} CER {m['cer']:.4f} | WER {m['wer']:.4f} | n={m['n_eval']} | best_cer={best:.4f}")
    print(f"best:   {best_path}\nresume: {resume_path}")
    for ref, hyp in m["samples"]:
        print(f"REF: {ref[:100]}\nHYP: {hyp[:100]!r}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="data dir (default: ./data/librispeech100 or $HOLO_DATA)")
    ap.add_argument("--train-file", default="libri_train.jsonl.zst")
    ap.add_argument("--heldout-file", default="libri_heldout.jsonl.zst")
    ap.add_argument("--frontend", default="hslspec", choices=["hsl", "mel", "hslwav", "hslspec"])
    ap.add_argument("--fuse", default="gate", choices=["add", "gate"], help="hslspec substrate fusion (gate=learnable, init 0.1)")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-cap", type=int, default=0, help="0 = full heldout; e.g. 400 for fast iteration")
    ap.add_argument("--lr-floor", type=float, default=0.05)
    ap.add_argument("--max-bytes", type=int, default=120000)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--specaug", type=int, default=0)
    ap.add_argument("--sortagrad", type=int, default=1)
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume-path", default=None)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--eval-only", default="")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    if args.smoke:
        args.dim, args.layers, args.heads, args.ff = 64, 2, 4, 256
        args.batch = args.eval_batch = 4
        args.steps, args.warmup, args.log_every, args.eval_every, args.eval_cap = 8, 2, 2, 4, 8
        args.max_bytes, args.limit_train, args.device = 0, 32, "cpu"
    run(args)


if __name__ == "__main__":
    main()
