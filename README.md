# HoLo-ToLk (STT) — an HSL-lens speech-to-text feasibility model

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!-- DOI badge: TODO. After you connect this repo to Zenodo and cut the first GitHub release,
     Zenodo mints a DOI. Paste the badge here, e.g.:
     [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
     and fill the same DOI into CITATION.cff and the BibTeX block below. Do NOT invent a number. -->

> 🇰🇷 이 프로젝트는 개인 시간에 독립적으로 연구·공개한 오픈 연구 산출물입니다.
> 🇬🇧 This is an independent, open research project — researched and released on personal time.

**Built on:** [`hsl-embedding`](https://github.com/Woojiggun/hsl-embedding) (the byte-signal substrate)
· [`hsl-embedding-zero`](https://github.com/Woojiggun/hsl-embedding-zero) — `pip install hsl-embedding-zero`
(the zero-parameter encoder this model feeds bytes through).

> **The name:** **HoLo-ToLk** is the **speech line** (*ToLk* reads as "talk", stylized to match the
> **HoLo-ZeRo / HoLo-AuTo** family — and *tolk* means "interpreter" in Scandinavian / Dutch, fitting
> for speech and, later, speech-translation). **This repo is the STT model, `HoLo-ToLk-STT`.** The
> **TTS** and **unified** models will be released as **separate sibling repos** (`HoLo-ToLk-TTS`,
> and a unified STT+TTS model) as they firm up.

---

## What it is

A **works-demonstration** that the zero-parameter byte-signal substrate (`hsl-embedding-zero`),
**plus a model-side spectral LENS**, wired into a plain character-CTC ASR baseline, actually
transcribes speech — **and that the lens is what makes it work.**

The honest one-liner:

> Pure `hsl-embedding-zero` fed straight to a char-CTC ASR baseline is **weak** (CER ~0.67).
> Adding a model-side **spectral lens** (log-mel + a learnable gated fusion over the lossless HSL
> substrate) **flips it**: CER **0.194**, beating the mel-spectrogram baseline (**0.213**) in the
> **same setup**, confirmed across **4 seeds** (hslspec-gate < mel on every seed; mean **0.193**,
> range 0.002). HSL is a **raw substrate** — alone it underperforms, but with the right lens its
> performance changes. This is a **feasibility / works demonstration** from an independent
> single-GPU project (8 kHz, no language model, char-CTC) — readable but rough in absolute terms,
> **NOT** a competitive/SOTA ASR product.

## The result (money chart)

Identical char-CTC / Pre-LN transformer / SortaGrad pipeline — the **only** thing that changes is
the front-end. LibriSpeech train-clean-100, 8 kHz, **no language model**, greedy CTC decode.
Held-out = dev-clean (n = 2642).

| front-end | what it feeds the encoder | held-out CER |
|---|---|---:|
| `hsl` | HSL(audio) derivative substrate **only** | **0.673** (floor) |
| `hslwav` | substrate + raw time-domain (wav / Δ / energy) | 0.709 |
| `mel` | log-mel spectrogram **only** | **0.213** (ceiling, this setup) |
| **`hslspec` + gated fusion** | **substrate + log-mel lens (learnable gate)** | **0.194** ✅ |

- Adding **more time-domain** features (`hslwav`) does **not** help — the active ingredient is
  specifically the **time→frequency transform**, not "more features".
- The fusion **gate** (the substrate's contribution) climbed **0.1 → 0.26** over training: the
  lossless HSL substrate **genuinely complements** magnitude-mel (mel discards phase / fine-timing;
  the substrate keeps it). It is an honest built-in diagnostic — ≈0 would mean "substrate unused".

### Multi-seed (the comparison is not seed noise)

Eval-only on the 6 saved checkpoints, full held-out (n = 2642):

| seed | `hslspec` + gate (CER) | `mel` (CER) |
|---|---:|---:|
| 0 | **0.194** | 0.213 |
| 1 | **0.194** | 0.208 |
| 2 | **0.194** | 0.342 \* |
| 3 | **0.192** | 0.210 |
| **mean** | **0.193** (range 0.002) | — |

`hslspec` + gate is tight across seeds (range 0.002 — not seed noise) and **beats `mel` on every
seed**. \*`mel` seed 2 was an unstable run; `hslspec` still beats the **good** `mel` seeds
(0.208 / 0.210), so the win does **not** hinge on that outlier.

## How it works

```
 mu-law audio bytes (8 kHz)
        │
        ├─────────────────────────────┐
        │                             │
   HSL substrate                 spectral LENS
   hsl-embedding-zero            log-mel (STFT, fixed filterbank)
   (frozen 27-D, 0 params)            │
        │                             │
   ConvSub (stride 160)          mel projection
        │                             │
        └──────── gated fusion ───────┘     lens + gate · substrate   (gate init 0.1, learnable)
                       │
              Pre-LN Transformer encoder  (8 layers, dim 384)
                       │
                 char-CTC head  →  greedy decode  →  text
```

- **Substrate** — every mu-law byte goes through the frozen, zero-parameter `hsl-embedding-zero`
  encoder: a multi-order **derivative** representation (Gray-code change-rate Δ, 2nd-order Δ²,
  boundary, exact bit-FFT, phase) — 27 dims/byte, lossless. It captures *change*, but a per-byte op
  cannot reach across samples to expose acoustic frequency — hence the lens.
- **Lens** — a **model-side, parameter-free** spectral feature lens (the audio sibling in the
  `holo_core.hsl_signal` lens family): log-mel via a fixed STFT + mel filterbank. It supplies the
  **time→frequency** transform the per-byte substrate lacks. In this repo it is **vendored inline**
  in `asr_lens.py` (torchaudio `MelSpectrogram` + a learnable scalar gate), so the script runs from
  `pip install hsl-embedding-zero` alone — no extra package required.
- **Fusion** — `lens + gate · substrate` with a learnable scalar gate (init 0.1). Training starts on
  the proven spectral lens and **adds** the lossless substrate only where it reduces loss. Additive,
  never a replacement — the lossless core is preserved.
- **Head** — a plain character CTC over `[space] a–z '` (29 symbols incl. blank), greedy decode, no
  language model.

## Install

```bash
pip install hsl-embedding-zero zstandard numpy torch torchaudio
```

`hsl-embedding-zero` pulls in `hsl-embedding` (the substrate) and `torch`. `torchaudio` provides the
mel front-end. That's the entire runtime dependency set for `asr_lens.py`.

## Reproduce

Paths below use a **relative** `./data/librispeech100`. Set `--data <dir>` (or the `HOLO_DATA`
env var) to point anywhere. A CUDA GPU is recommended for training; eval runs on CPU.

**1. Build the data** (rebuilds LibriSpeech-100 locally — see *Data* below; not redistributed here):

```bash
pip install -U datasets soundfile librosa zstandard numpy
python collect_libri.py --out ./data/librispeech100
# -> ./data/librispeech100/libri_train.jsonl.zst   (~28.5k clips)
#    ./data/librispeech100/libri_heldout.jsonl.zst (~2.6k clips, dev-clean)
```

**2. Train the winning front-end** (`hslspec` + gated fusion, seed 0):

```bash
python asr_lens.py --frontend hslspec --fuse gate \
  --data ./data/librispeech100 \
  --dim 384 --layers 8 --heads 6 --ff 1536 \
  --batch 24 --lr 4e-4 --max-bytes 120000 --steps 12000 --seed 0
```

Swap `--frontend` to reproduce the rest of the money chart:
`mel` (ceiling) · `hsl` (floor) · `hslwav` (time-domain, inert) · `hslspec --fuse gate` (**the winner**).

**3. Eval only** (load a frozen checkpoint, score the full held-out set):

```bash
python asr_lens.py --eval-only ./data/librispeech100/asr_lens_best_hslspec_gate.pt \
  --data ./data/librispeech100
# prints CER / WER over dev-clean (n=2642) + a few REF/HYP samples
```

## Data

- **LibriSpeech `train-clean-100`** (train) + **`dev-clean`** (held-out), license **CC BY 4.0**.
  Official splits ⇒ speaker/chapter isolation by construction.
- Each clip is resampled to **8 kHz** and **mu-law** encoded to bytes; rows are JSON Lines, zstd-
  compressed: `{"text", "ulaw_b64", "sr": 8000, ...}`.
- **The audio is NOT redistributed in this repo.** `collect_libri.py` **rebuilds** it locally by
  streaming from the Hugging Face `openslr/librispeech_asr` dataset.

## Checkpoint

The frozen **seed-0** winner (`hslspec` + gate, step 12k, **CER 0.194**) is the `asr_lens_best_hslspec_gate.pt`
file (~200 MB). It is hosted on the Hugging Face model repo, not in git:

- 🤖 **Model:** https://huggingface.co/ggunio/HoLo-ToLk-STT *(weights uploaded there; see [`MODEL_CARD.md`](MODEL_CARD.md))*

Download it next to your data dir and point `--eval-only` at it (see *Reproduce → 3*). The
checkpoint stores its own `config` and `vocab`, so `asr_lens.py` rebuilds the exact model on load.

## Try it

A minimal Hugging Face **Gradio Space** lives in [`space/`](space/): **record your own voice or
upload audio** → resample to 8 kHz mu-law → load the frozen checkpoint → char-CTC greedy decode →
transcript. **English only.** It carries an honest banner that the output is a **rough feasibility
demo** (8 kHz, no LM) — a clear, slowly-spoken English sentence gives the most legible output.

## Scope (honest)

HoLo-ToLk-STT is a **feasibility / works demonstration** from an independent, single-GPU project — a
possibility-proof, **not** a benchmark-beating or production ASR system.

- **English only** (trained on LibriSpeech English read speech) — other languages will not work.
- **8 kHz · no language model · char-CTC** ⇒ the transcript is **readable but garbled** in absolute
  terms. Do not expect clean text.
- The result is the **controlled comparison** — substrate + spectral lens **>** mel **in the same
  setup, multi-seed** — **not** a competitive ASR product, and **not** a claim of SOTA or general
  superiority.
- HSL alone is a **raw substrate**: fed straight to this baseline it underperforms (CER ~0.67). The
  point is that the *right lens* changes that — the substrate is necessary but not sufficient for a
  signal modality, and the lens supplies the missing time→frequency axis.

## Status & roadmap (honest)

**HoLo-ToLk** is the **speech line**; this repo (`HoLo-ToLk-STT`) is its **STT model**. The TTS and
unified models are **separate sibling releases**, coming as they firm up:

- ✅ **STT — `HoLo-ToLk-STT` (this repo):** HSL substrate + spectral lens beats the mel baseline, multi-seed confirmed (above).
- ✅ **TTS — `HoLo-ToLk-TTS` (separate sibling repo, coming):** text → HSL (no tokenizer) → AR mel decoder + guided attention → HiFi-GAN produces a **natural-sounding voice** (held-out mel-L1 ≈ 0.30, single seed). **Multi-seed firming is in progress.**
- 🚧 **Unified STT + TTS (separate sibling release, in progress):** folding both directions into ONE model is the hard open challenge. Keeping TTS free-run stable inside a shared model is genuinely difficult and **not solved yet** — but it is actively being worked on. **Not abandoned.**

Released honestly as the line develops — a possibility-proof, not a finished product.

## License & citation

**MIT License — © 2026 Jinhyun Woo (ggunio5782@gmail.com).**
Free to use, modify, and **distribute, including commercially** — the only condition is keeping the
copyright notice and attribution to **Jinhyun Woo**. See [LICENSE](LICENSE).

```bibtex
@software{woo_holotolk_stt_2026,
  author = {Jinhyun Woo},
  title  = {HoLo-ToLk (STT): an HSL-lens speech-to-text feasibility model},
  year   = {2026},
  url    = {https://github.com/Woojiggun/HoLo-ToLk-STT}
  % doi  = {10.5281/zenodo.XXXXXXX}   % TODO: add after the first Zenodo release
}
```
