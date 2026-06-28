---
title: HoLo-ToLk (STT)
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# HoLo-ToLk (STT) — speech-to-text (rough feasibility demo)

A minimal Gradio demo of **HoLo-ToLk-STT**: **record your own voice or upload audio** → resample to
8 kHz mu-law → the frozen `hslspec` + gated-fusion checkpoint → char-CTC greedy decode → transcript.

> ⚠️ **Rough feasibility demo — NOT a usable transcriber. English only** (LibriSpeech read speech).
> Expect **garbled output**: it runs at **8 kHz** (downsampled, high frequencies lost), **no language
> model**, **character-level CTC** (no spell/word correction), trained on ~100h on a **single GPU**.
> The point is the controlled result — an HSL substrate **+ spectral lens** beats a mel baseline
> *in the same setup* (CER **0.194** vs **0.213**, multi-seed) — **not** the transcript itself.
> A **clear, slowly-spoken English sentence** gives the most legible output.

**Input:** records from your **microphone** or accepts an **uploaded file** (any sample rate; it is
resampled to 8 kHz internally). **Language: English only** — other languages will not work.

## Where this sits

**HoLo-ToLk** is the **speech line**; this is its **STT model** (`HoLo-ToLk-STT`). The **TTS**
(`HoLo-ToLk-TTS`) and a **unified** STT+TTS model are **separate sibling releases**, coming as they
firm up.

- Code + full writeup: **https://github.com/Woojiggun/HoLo-ToLk-STT**
- Model weights: **https://huggingface.co/ggunio/HoLo-ToLk-STT**
- Substrate: **https://github.com/Woojiggun/hsl-embedding-zero**

The checkpoint is pulled from the HF model repo `ggunio/HoLo-ToLk-STT` at startup
(`asr_lens_best_hslspec_gate.pt`, ~200 MB). Set the `HOLOTOLK_CKPT` env var to a local `.pt` path
to use your own.

MIT © 2026 Jinhyun Woo.
