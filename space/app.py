"""HoLo-ToLk (STT) - Gradio Space (rough feasibility STT demo).

Record your own voice OR upload audio -> resample to 8 kHz mu-law -> load the frozen `asr_lens`
checkpoint (hslspec + gated fusion, seed 0) -> char-CTC greedy decode -> transcript.

HONEST CAVEAT: this is a feasibility / works demonstration. ENGLISH ONLY (trained on LibriSpeech
English read speech). The output is readable but ROUGH/garbled in absolute terms (8 kHz, no language
model, character-level CTC, ~100h single-GPU training). It is NOT a usable transcriber. The point of
the project is the controlled comparison (HSL substrate + spectral lens > mel, same setup,
multi-seed) -- see https://github.com/Woojiggun/HoLo-ToLk-STT .

This Space ships its own copy of `asr_lens.py` (the frozen model definition). The checkpoint is
pulled from the HF model repo `ggunio/HoLo-ToLk-STT` (override with the HOLOTOLK_CKPT env var to use a
local path).
"""
import os

os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import torch
import gradio as gr

# Frozen model definition (CharASR, mu_law_decode, greedy_decode, normalize_text, ...).
import asr_lens as A

MODEL_REPO = os.environ.get("HOLOTOLK_REPO", "ggunio/HoLo-ToLk-STT")
CKPT_NAME = os.environ.get("HOLOTOLK_CKPT_NAME", "asr_lens_best_hslspec_gate.pt")
LOCAL_CKPT = os.environ.get("HOLOTOLK_CKPT", "")          # set to a local .pt path to skip the HF download
TARGET_SR = 8000

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_ckpt() -> str:
    if LOCAL_CKPT and os.path.exists(LOCAL_CKPT):
        return LOCAL_CKPT
    # Pull the frozen seed-0 checkpoint from the Hugging Face model repo.
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MODEL_REPO, filename=CKPT_NAME)


def _load_model():
    global _MODEL
    if _MODEL is None:
        path = _resolve_ckpt()
        _MODEL, _ck = A.load_model_from_checkpoint(path, _DEVICE)
        _MODEL.eval()
    return _MODEL


def _to_mulaw_bytes(wav: np.ndarray, sr: int) -> np.ndarray:
    """float waveform [-1,1] @ sr -> 8 kHz mu-law byte ids [0,255]."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:                                     # stereo -> mono
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:                                       # int-range input -> normalize to [-1,1]
        wav = wav / peak
    if sr != TARGET_SR and wav.size:                     # resample to 8 kHz
        import torchaudio
        t = torch.from_numpy(wav).float().unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, TARGET_SR)
        wav = t.squeeze(0).numpy()
    return _mulaw_encode(wav)


def _mulaw_encode(x, mu=255):
    """waveform float [-1,1] -> mu-law byte ids [0,255] (matches collect_libri.py)."""
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.clip((y + 1.0) / 2.0 * mu + 0.5, 0, mu).astype(np.uint8)


@torch.no_grad()
def transcribe(audio):
    if audio is None:
        return "(no audio) — upload a file or record a clip."
    sr, wav = audio                                      # gradio Audio(type="numpy") -> (sample_rate, np.ndarray)
    ulaw = _to_mulaw_bytes(wav, sr)
    if ulaw.size == 0:
        return "(empty audio)"
    model = _load_model()
    ids = torch.tensor(ulaw.astype(np.int64), dtype=torch.long, device=_DEVICE).unsqueeze(0)
    logp, in_len = model(ids, [int(ulaw.size)])
    hyp = A.greedy_decode(logp, in_len)[0]
    return hyp if hyp.strip() else "(silence / nothing decoded)"


BANNER = """
# HoLo-ToLk (STT) — speech-to-text (rough feasibility demo)

> ⚠️ **Please read before testing — this sets your expectations.**
> A **feasibility / works demonstration, NOT a usable transcriber.** The model was trained on **clean,
> read-aloud English sentences** (LibriSpeech audiobooks) at **8 kHz**, with **no language model** and a
> character-level CTC head.
>
> **A single word or casual speech — e.g. saying "hello" into a laptop mic — is _out-of-distribution_ and
> looks much worse than the headline number.** Short, spontaneous, room-mic audio is the hardest case for it.
>
> **For representative output:** click an **Example** below (real LibriSpeech clips — the kind of audio it
> was trained on), or **read a full English sentence aloud, clearly and slowly** (like narrating a book).
> Even then it stays **readable-but-rough by design.**
>
> **What matters is the controlled comparison** — HSL substrate **+ spectral lens beats the mel baseline in
> the same setup, multi-seed (CER 0.194 vs 0.213)** — **not the transcript itself.**

Substrate: [`hsl-embedding-zero`](https://github.com/Woojiggun/hsl-embedding-zero) (zero-parameter byte encoder).
Details + code: **https://github.com/Woojiggun/HoLo-ToLk-STT**
"""

with gr.Blocks(title="HoLo-ToLk (STT)") as demo:
    gr.Markdown(BANNER)
    with gr.Row():
        inp = gr.Audio(
            sources=["microphone", "upload"],            # record your own voice OR upload a file
            type="numpy",
            label="English speech — record your voice or upload (any rate; resampled to 8 kHz)",
        )
    gr.Examples(
        examples=[["examples/sample1.wav"], ["examples/sample2.wav"], ["examples/sample3.wav"]],
        inputs=inp,
        label="In-domain LibriSpeech examples (the audio this model was trained on) - click one, then Transcribe",
    )
    btn = gr.Button("Transcribe", variant="primary")
    out = gr.Textbox(label="Transcript (English, greedy char-CTC, no LM — expect rough output)", lines=4)
    btn.click(transcribe, inputs=inp, outputs=out)
    gr.Markdown(
        "**Example reference transcripts** (the model's output should be *readable-but-rough*, not perfect "
        "— and a casual mic clip will be far worse than these):\n\n"
        "1. _he was in a fevered state of mind owing to the blight his wife's action threatened to cast upon his entire future_\n"
        "2. _he would have to pay her the money which she would now regularly demand or there would be trouble it did not matter what he did_\n"
        "3. _hurstwood walked the floor mentally arranging the chief points of his situation_"
    )
    gr.Markdown(
        "_Model: `hslspec` + gated fusion, seed 0 (CER 0.194 on LibriSpeech dev-clean, English). "
        "Rough feasibility demo. CC BY-NC 4.0 (non-commercial) © 2026 Jinhyun Woo._"
    )

if __name__ == "__main__":
    demo.launch()
