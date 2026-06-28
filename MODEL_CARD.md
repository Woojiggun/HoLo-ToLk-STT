---
license: mit
language:
  - en
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
  - speech-to-text
  - asr
  - ctc
  - tokenizer-free
  - byte-native
  - hsl
  - feasibility
datasets:
  - openslr/librispeech_asr
metrics:
  - cer
  - wer
---

# HoLo-ToLk (STT) — an HSL-lens speech-to-text feasibility model

A **works-demonstration** that the zero-parameter byte-signal substrate
[`hsl-embedding-zero`](https://github.com/Woojiggun/hsl-embedding-zero), plus a **model-side
spectral lens** (log-mel + a learnable gated fusion over the lossless HSL substrate), wired into a
plain character-CTC ASR baseline, transcribes speech — **and that the lens is what makes it work.**

- **Code / reproduce:** https://github.com/Woojiggun/HoLo-ToLk-STT
- **Author:** Jinhyun Woo (ggunio5782@gmail.com) · **License:** MIT

## The honest claim

> Pure `hsl-embedding-zero` fed straight to a char-CTC ASR baseline is **weak** (CER ~0.67). Adding a
> model-side spectral lens **flips it**: CER **0.194**, beating the mel-spectrogram baseline (**0.213**)
> in the **same setup**, confirmed across **4 seeds** (hslspec-gate < mel on every seed; mean **0.193**,
> range 0.002). HSL is a raw substrate — alone it underperforms, but with the right lens its performance
> changes. This is a **feasibility / works demonstration** (8 kHz, no language model, char-CTC) —
> readable but rough in absolute terms, **NOT** a competitive/SOTA ASR product.

| front-end | held-out CER |
|---|---:|
| `hsl` (substrate only) | 0.673 |
| `hslwav` (substrate + time-domain) | 0.709 |
| `mel` (spectral only) | 0.213 |
| **`hslspec` + gated fusion** | **0.194** |

Multi-seed (`hslspec`+gate vs `mel`): seeds 0–3 = **0.194 / 0.194 / 0.194 / 0.192** vs
0.213 / 0.208 / 0.342\* / 0.210 (\*mel seed 2 unstable; hslspec still beats the good mel seeds).
Fusion gate learned **0.1 → 0.26**.

## Intended use

- **Research / educational** demonstration that a lossless byte substrate + a model-side spectral
  lens **>** mel in a controlled char-CTC ASR setup; and a worked example of feeding bytes through
  the zero-parameter HSL encoder.
- A starting point for experiments with tokenizer-free / byte-native signal front-ends.

**Not** intended for production transcription, captioning, or any use needing accurate text.

## How it works

mu-law audio bytes (8 kHz) → **two paths**: (a) the frozen 27-D `hsl-embedding-zero` derivative
substrate → ConvSub; (b) a parameter-free **log-mel** spectral lens → projection. Fused as
`lens + gate · substrate` (learnable gate, init 0.1) → Pre-LN Transformer encoder (8 layers, dim
384) → char-CTC head (29 symbols) → greedy decode, **no language model**. See the repo README and
`asr_lens.py` for the exact architecture.

## Training data

**LibriSpeech `train-clean-100`** (≈28.5k clips, 100h) for training; **`dev-clean`** (≈2.6k) held
out — official splits, license **CC BY 4.0**. Audio resampled to **8 kHz**, mu-law encoded.
The dataset is **not redistributed**; rebuild it with `collect_libri.py` (streams from
`openslr/librispeech_asr`).

## Training procedure

char-CTC, Pre-LN Transformer, SortaGrad short-clips-first. `dim 384 / layers 8 / heads 6 / ff 1536`,
batch 24, AdamW lr 4e-4 (cosine, 800-step warmup), 12k steps, grad-clip 5.0, seed 0. The reported
checkpoint is the best held-out CER over training.

## Evaluation

CER / WER on the full `dev-clean` held-out set (n = 2642), greedy CTC decode, no LM.
Reported: **CER 0.194 / WER 0.535** for `hslspec` + gate (seed 0).

## Limitations

- **8 kHz / no LM / char-CTC** ⇒ output is **readable but garbled**; expect substitutions and
  malformed words. Not usable where accurate text matters.
- English read-speech (LibriSpeech) only; not evaluated on other languages, noisy/far-field audio,
  spontaneous speech, or other sampling rates.
- The contribution is a **controlled comparison** (substrate + lens > mel, same setup, multi-seed),
  **not** state-of-the-art accuracy.

## Files / checkpoint to upload

Upload the frozen **seed-0** winner to this model repo:

- **`asr_lens_best_hslspec_gate.pt`** — `hslspec` + gate, step 12k, seed 0, **CER 0.194** (~200 MB).
  Self-contained: stores its own `config` + `vocab`, so `asr_lens.py` rebuilds the model on load.

Load & score:

```bash
pip install hsl-embedding-zero zstandard numpy torch torchaudio
python asr_lens.py --eval-only asr_lens_best_hslspec_gate.pt --data ./data/librispeech100
```

## Acknowledgments

Independent research, developed in collaboration with AI assistants — **Claude Code** (Anthropic)
and **Codex**. The HSL work and experimental direction are the author's; the tools assisted with
engineering and review.

## Citation

```bibtex
@software{woo_holotolk_stt_2026,
  author = {Jinhyun Woo},
  title  = {HoLo-ToLk (STT): an HSL-lens speech-to-text feasibility model},
  year   = {2026},
  doi    = {10.5281/zenodo.21004333},
  url    = {https://github.com/Woojiggun/HoLo-ToLk-STT}
}
```
