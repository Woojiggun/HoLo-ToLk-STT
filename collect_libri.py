"""collect_libri.py - build the LibriSpeech train-clean-100 -> 8 kHz mu-law dataset for HoLo-ToLk-STT.

Standard ASR benchmark (everyone reports WER). Uses the official splits:
  train   = train.clean.100  (100h, 251 speakers)
  heldout = dev-clean        (validation)
=> speaker/chapter isolation by construction. 8 kHz mu-law (consistent with the model).
LibriSpeech is CC BY 4.0. We do NOT redistribute the audio -- this script rebuilds it locally
by streaming from the Hugging Face `openslr/librispeech_asr` dataset.

The audio is streamed and written incrementally (constant memory), so the full train-clean-100
set never has to be held in RAM at once.

Output rows are JSON Lines, zstd-compressed:
  {"text": <transcript>, "sr": 8000, "dur": <sec>, "n": <num bytes>, "spk": <id>, "ulaw_b64": <base64 mu-law>}

Run (a GPU is not needed for this step; a fast download / SSD is):
  pip install -U datasets soundfile librosa zstandard numpy
  python collect_libri.py --out ./data/librispeech100
"""
import os, sys, json, base64, argparse, time
import numpy as np


def mulaw_encode(x, mu=255):
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.clip((y + 1.0) / 2.0 * mu + 0.5, 0, mu).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="openslr/librispeech_asr")
    ap.add_argument("--config", default="clean")
    ap.add_argument("--train-split", default="train.100")
    ap.add_argument("--heldout-split", default="validation")        # dev-clean
    ap.add_argument("--per-train", type=int, default=30000)         # train.100 = ~28.5k
    ap.add_argument("--per-heldout", type=int, default=2800)        # dev-clean = ~2.7k
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--min-sec", type=float, default=0.8)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--out", default="./data/librispeech100")
    args = ap.parse_args()
    from datasets import load_dataset, Audio
    import zstandard as zstd
    os.makedirs(args.out, exist_ok=True)

    for split, cap, name in ((args.train_split, args.per_train, "train"),
                             (args.heldout_split, args.per_heldout, "heldout")):
        print(f"[libri] {name}: {args.dataset}/{args.config}/{split} streaming...", flush=True)
        ds = load_dataset(args.dataset, args.config, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=args.sr))
        path = os.path.join(args.out, f"libri_{name}.jsonl.zst")
        cnt = 0; skip = 0; t0 = time.time()
        with open(path, "wb") as f:                              # stream write (constant memory)
            with zstd.ZstdCompressor(level=10).stream_writer(f, closefd=False) as w:
                for ex in ds:
                    txt = (ex.get("text") or ex.get("sentence") or "").strip()
                    if not txt:
                        skip += 1; continue
                    arr = np.asarray(ex["audio"]["array"], dtype=np.float32)
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                    dur = len(arr) / args.sr
                    if dur < args.min_sec or dur > args.max_sec:
                        skip += 1; continue
                    ul = mulaw_encode(arr)
                    rec = {"text": txt.lower(), "sr": args.sr, "dur": round(dur, 2), "n": int(len(ul)),
                           "spk": str(ex.get("speaker_id", "")),
                           "ulaw_b64": base64.b64encode(ul.tobytes()).decode("ascii")}
                    w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
                    cnt += 1
                    if cnt % 2000 == 0:
                        print(f"  {name} {cnt} (skip {skip}, {time.time()-t0:.0f}s)", flush=True)
                    if cnt >= cap:
                        break
        print(f"[libri] wrote {path} ({cnt} rows, {os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    print("[libri] done.")


if __name__ == "__main__":
    main()
