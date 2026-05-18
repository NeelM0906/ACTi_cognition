# TRIBE — Research Engineer Brief

> Internal handoff for an engineer tasked with (1) understanding the offering, (2) speeding it up, (3) improving model quality. Paper to follow separately.

---

## 1. What it is, in one paragraph

**TRIBE v2** (Meta FAIR, d'Ascoli et al., 2026) is a multimodal **fMRI brain-encoding** foundation model. Given a naturalistic stimulus (video, audio, **or text**), it predicts the BOLD response that a typical human cortex would produce, expressed as activation values over **~20k vertices on the `fsaverage5` cortical surface**, shifted 5s back to compensate for hemodynamic lag. Weights are open on HF Hub at `facebook/tribev2` under CC-BY-NC-4.0. Our ACTi deployment wraps it in a **two-stage pipeline**: (a) run TRIBE → produce a cortical activation heatmap PNG, (b) feed PNG + original stimulus to a vision-LLM that returns a **behavioral interpretation** in Markdown. That second stage is *our addition*, not part of the paper.

**Service surface:** `POST /analysis` on `https://acti.cognition.ngrok.pro` (see `/home/ripper/ACTi_cognition/API.md`).

---

## 2. The full pipeline (text input — the path your engineer will optimize)

```
user text
  │
  ▼ tribev2.demo_utils.TextToEvents.get_events
  │
  ├── 1. gTTS (Google Text-to-Speech) ── synth audio.mp3
  │
  ▼ tribev2.demo_utils.get_audio_and_text_events
  │
  ├── 2. ExtractAudioFromVideo  (no-op for audio input)
  ├── 3. ChunkEvents (Audio,  30–60 s windows)
  ├── 4. ChunkEvents (Video,  no-op)
  ├── 5. ExtractWordsFromAudio  ─►  whisperx large-v3 + WAV2VEC2 alignment
  ├── 6. AddText                 (full transcript)
  ├── 7. AddSentenceToWords      (sentence boundaries)
  ├── 8. AddContextToWords       (up to 1024-token window per word)
  └── 9. RemoveMissing
  │
  ▼ TribeModel.predict  (this is where the GPU starts to matter)
  │
  ├── 10. Per-modality feature extraction (all loaded lazily, results cached on disk):
  │     • text   : HuggingFace LLM hidden states (unsloth/Llama-3.2-3B, 6 layers, 2 Hz)
  │     • audio  : Wav2Vec-BERT 2.0 (2 layers, 2 Hz)        ← always runs for text-input
  │     • video  : (skipped — only fired if a video path is provided)
  │
  ├── 11. FmriEncoderModel forward:
  │     projectors → combiner → transformer (depth=8, hidden=1152)
  │     → low-rank head (2048) → subject-layer Linear → AdaptiveAvgPool1d
  │     → (n_timesteps, n_vertices=20484)  on fsaverage5
  │
  ▼ tribev2.plotting.PlotBrain.plot_timesteps  (PyVista, off-screen)
  │
  └── 12. PNG on disk, ~MB
  │
  ▼ ACTi LLM stage
  │
  └── 13. Vision-LLM via OpenRouter:
        SYSTEM_PROMPT (cognitive-neuroscience translator)
        + USER_TEMPLATE(stimulus_description, image_b64)
        → behavioral analysis markdown (~5–10 KB)
```

Steps 1–9 live in `tribev2/eventstransforms.py` + `tribev2/demo_utils.py`. Steps 10–11 in `tribev2/model.py`. Step 12 in `tribev2/plotting/`. Step 13 in `app.py`.

---

## 3. Model architecture (FmriEncoder — `tribev2/model.py`)

```
                    ┌──────────────┐
text features ──►│  MLP project │──┐
(L×D per word)    └──────────────┘  │
                                    ├─►  concat ─► MLP combiner
audio features ──►│  MLP project │──┤            (hidden=1152)
(L×D per frame)   └──────────────┘  │              │
                                    │              ▼
video features ──►│  MLP project │──┘   + time positional embedding
(L×D per frame)   └──────────────┘            │
                                              ▼
                                   TransformerEncoder
                                   (depth=8, hidden=1152,
                                    causal=False, attn+ff dropout)
                                              │
                                              ▼
                                   Linear  low_rank_head: 1152→2048
                                              │
                                              ▼
                                   SubjectLayers
                                   (per-subject Linear, subject_dropout=0.1;
                                    inference uses "average subject")
                                              │
                                              ▼
                                   AdaptiveAvgPool1d → n_timesteps
                                              │
                                              ▼
                                  (n_timesteps, n_vertices≈20484)
```

Key configuration (see `tribev2/grids/defaults.py`):
- `hidden=1152`, `encoder.depth=8`, `low_rank_head=2048`
- `layer_aggregation="cat"`, `extractor_aggregation="cat"` (concatenates layers within a modality, then concatenates modalities)
- `modality_dropout=0.3` (training only) — model is trained to handle missing modalities, which is why text-only input still works at inference
- `subject_embedding=False`, `time_pos_embedding=True`
- Surface projection: `fsaverage5` mesh, `ball` aggregation, radius 3 mm
- TR (sampling) = 2 Hz (i.e. 500 ms), output offset = 5 s past (hemodynamic correction)

### Feature extractors (the heavy parts)

| Modality | Model | Layers | Freq | Notes |
|---|---|---|---|---|
| Text | `unsloth/Llama-3.2-3B` (override in `app.py`; default in paper: `meta-llama/Llama-3.2-3B`) | `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` (6 evenly-spaced) | 2 Hz | Contextual, 1024-token window per word |
| Audio | `Wav2VecBert` (W2V-BERT 2.0) | `[0.75, 1.0]` | 2 Hz | Always runs for text-input path because of TTS detour |
| Video (img) | `facebook/dinov2-large` | `2/3` | 2 Hz | Image features |
| Video | `facebook/vjepa2-vitg-fpc64-256` | `[0.75, 1.0]` | 2 Hz | 4-s clips |

Features are **cached on disk** keyed by content hash under `cache/neuralset.extractors.*/...` — once a stimulus has been processed, repeat calls reuse features. Sizes today: text-feature cache 692 MB, audio 174 MB, video 122 MB.

---

## 4. Training (paper + defaults.py)

- **Datasets** (4 large naturalistic fMRI corpora):
  - `Algonauts2025Bold` (movie-watching)
  - `Wen2017` (movie-watching)
  - `Lahner2024Bold` (BOLD5000-like)
  - `Lebel2023Bold` (story-listening)
- 15 epochs, Adam, OneCycleLR (max_lr=1e-4, pct_start=0.1), MSE loss
- Metrics: per-vertex Pearson r, subject-grouped Pearson, top-1 retrieval
- Train on Slurm, 1 GPU/job for feature extractors, 1 GPU for the encoder
- `duration_trs=100`, `batch_size=8`, `num_workers=20`
- "Average subject" head is what we ship at inference

---

## 5. The ACTi LLM stage (our addition, not the paper)

`app.py::analyze_brain_image` + `analyze_text_endpoint`.

**Model selection (internal — do not expose externally per our "no LLM identity in dev artifacts" rule):**
- Currently configured via `OPENROUTER_ANALYSIS_MODEL` in `.env`
- Default in code: `anthropic/claude-opus-4`
- Override in `analyze_text_endpoint`: `xiaomi/mimo-v2.5-pro`

**Prompt design (`app.py` lines 142–158):**
- System prompt frames the model as a "cognitive neuroscience translator"
- Output structure asked: (1) temporal arc, (2) peak activation moment + mechanism, (3) brain regions in plain language, (4) user-behavior prediction
- Image is sent base64-encoded inline; `max_tokens=25000`

This is the part most easily tuned without retraining TRIBE — prompt engineering, model selection, structured output, or replacing the vision-LLM call with a domain-specific finetune.

---

## 6. Where the time goes (latency budget)

Measured today on one 1.4 KB stimulus, 2× RTX PRO 6000 (TRIBE alone, no vLLM coexistence): **227 s total**.

| Stage | Approx. share | Notes |
|---|---|---|
| gTTS round-trip | 1–3 s | Network-bound, Google |
| whisperx large-v3 transcription | 30–90 s | GPU; CUDA OOM if VRAM contested |
| Llama-3.2-3B hidden-state extraction (6 layers, contextual) | 30–60 s | GPU; ~6 GB fp16 |
| Wav2Vec-BERT 2.0 features (fires unnecessarily for text input) | 10–25 s | GPU |
| FmriEncoder forward | <2 s | Tiny (~16M params) |
| PyVista rendering of N timesteps × M views | 20–40 s | CPU-bound, off-screen; DPI 150 for n<15 |
| OpenRouter vision-LLM call | 30–90 s | Network + remote inference |

> Per `API.md` we tell devs **10–15 min**, which is conservative for long stimuli + repeated views. Single-shot, single-view, short stimulus is much faster.

**Hardware footprint (live, idle):** TRIBE main process holds **1.91 GB on GPU 0**, GPU 1 is essentially empty. Active inference spikes peak VRAM significantly because whisperx + Llama-3.2-3B both get loaded.

**Concurrency:** hard global lock (`threading.Lock` in `app.py`). One request at a time, sequentially. There is no batching.

---

## 7. Speed-up levers (ranked by expected impact ÷ effort)

| # | Change | Where | Expected win | Risk |
|---|---|---|---|---|
| 1 | **Skip gTTS + whisperx for text input.** Synthesize word-timing events directly from text (~average word duration, optionally rate-modulated by sentence punctuation). The TTS→ASR detour exists because TRIBE was trained on audio with word-aligned timings — but the timings are coarse (2 Hz) and well-approximated by deterministic word-rate. | New transform in `tribev2/eventstransforms.py`, drop-in for `TextToEvents`. | **~30–50 % faster for text path**, removes the whisperx OOM failure mode entirely. | Slight distribution shift vs. training. Quantify Pearson delta on held-out text stimuli before shipping. |
| 2 | **Quantize Llama-3.2-3B to int4** (GPTQ/AWQ) for hidden-state extraction. We only read intermediate states, not generate. | `data.text_feature.model_name` override + `bitsandbytes`/`auto-gptq` load path in `neuralset.extractors.text.HuggingFaceText`. | ~2× faster text features, ~4× less VRAM. | Verify hidden-state correlation against fp16 baseline > 0.99 per layer. |
| 3 | **Suppress audio feature extraction for text-input path.** Currently `get_audio_and_text_events` runs the audio chunker before whisperx, and downstream `Wav2VecBert` still fires on the synthesized audio. With (1), there is no audio to extract. | `tribev2/demo_utils.py::get_audio_and_text_events`, gated on `audio_only`. | 10–25 s per call. | Need to verify training-time alignment: model was trained with `modality_dropout=0.3` so zeroed audio should be fine, but confirm. |
| 4 | **Local vision-LLM** instead of OpenRouter. Stand up a local VLM (we already have vLLM infra). | `app.py::analyze_brain_image` switches base_url + model. | 30–90 s saved per call + zero egress + no API key. | Quality regression vs. frontier. Worth A/B'ing. |
| 5 | **Cheaper rendering**: switch to `nilearn` cortical-surface plotting for fast paths (`tribev2/plotting/cortical.py` already has a Nilearn backend; PyVista is in `cortical_pv.py`). Drop DPI to 100 across the board; cache rendered PNGs per (timesteps, view, cmap, stimulus_hash). | `app.py::run_inference`. | 10–25 s saved per call. | Slightly lower-fidelity images for the LLM consumer. |
| 6 | **Async pre-warm** of all feature extractors at process start, not lazily on first call. | `app.py::_load_globals`. | First-request latency cut by ~30 s. | One-time at boot. |
| 7 | **Batch multiple `/analysis` requests** behind a request queue instead of `threading.Lock`. Feature extractors are batch-friendly; render is per-result. | `app.py` request handling + FmriEncoder forward. | Throughput, not latency. | Requires API contract for queueing. |
| 8 | **Compile the FmriEncoder** with `torch.compile` (it's only ~16M params and the forward is the same shape every call). | `tribev2/demo_utils.py::from_pretrained`. | Marginal (<1 s) since forward is already tiny. | Negligible. |
| 9 | **Replace whisperx large-v3 with `distil-whisper-large-v3` or `faster-whisper`** for non-text input paths. | `tribev2/eventstransforms.py::ExtractWordsFromAudio._get_transcript_from_audio`. | 2–4× faster ASR for audio/video inputs. | Marginal WER cost. |

> **The single biggest unlock is lever 1.** A 1.4 KB stimulus does not need TTS+ASR. That's also where the **CUDA OOM bug we hit today** disappears for free.

---

## 8. Quality levers (ranked by expected impact)

| # | Change | Rationale |
|---|---|---|
| 1 | **Replace MSE loss** with a Pearson-correlation loss or a contrastive (info-NCE over vertex sequences) loss. fMRI signals are best evaluated by correlation, not L2; training objective should match. | The paper's primary metric is Pearson r — but trains on MSE. |
| 2 | **Per-vertex learned HRF** instead of fixed 5 s offset. Hemodynamic response varies across cortex (early visual is ~4 s, prefrontal ~6 s); a vertex-wise learnable lag improves alignment. | `tribev2/grids/defaults.py:68` `neuro_extractor.offset=5` is scalar. |
| 3 | **Subject-specific heads with cross-subject distillation** rather than only "average subject." Per-subject Pearson improvements in the paper are large — currently we sacrifice that for ease of inference. | `SubjectLayers` already exists; we just need to expose subject id at the API level + a default-average fallback. |
| 4 | **Stronger text backbone**: swap `Llama-3.2-3B` for `Llama-3.1-8B` or `Qwen3-7B` hidden states; layers/aggregation already configurable in `defaults.py`. | Brain-encoding work generally finds bigger LMs → better fits up to a point. |
| 5 | **Modern audio backbone**: try `Whisper-3-encoder` hidden states or `EnCodec`/`SoundStream` representations in place of Wav2Vec-BERT. | Wav2Vec-BERT is 2023; speech-rep field has moved. |
| 6 | **Use V-JEPA3 / DINOv3** for image and video as they're released; layer-selection is already configurable. | Same family, drop-in. |
| 7 | **Higher-resolution surface output**: `fsaverage5` (~20k vertices) → `fsaverage6` (~41k) or `fsaverage7` (~163k). Predictor head is the only thing that needs changing; encoder body is unchanged. | More spatial detail = sharper renderings for the LLM stage. |
| 8 | **Volumetric (MNI) head** in addition to surface — subcortical regions (amygdala, hippocampus) are absent on `fsaverage*`. The repo already has `plotting/subcortical.py` and `grids/run_subcortical.py`; this is partly built. | The behavioral analysis prompt explicitly references emotional/memory mechanisms that *live* in subcortex — fixing this would meaningfully improve downstream LLM interpretations. |
| 9 | **Adaptive temporal smoothing** — the `TemporalSmoothing` module supports a learnable Gaussian kernel; we currently leave it default. Tune `kernel_size`/`sigma` per modality. | Cheap experiment. |
| 10 | **Train on more / newer datasets**: HCP-MOVIE, NSD, Courtois NeuroMod, additional Algonauts editions. Codebase already supports multi-study mixing via `tribev2/studies/`. | Diminishing returns above ~4 studies in the paper, but worth testing. |
| 11 | **Improve the LLM-stage prompt**: structured-output schema (JSON, not freeform Markdown), region-name vocabulary control, calibration on a held-out set of known neuroscience interpretations. | Decouples ACTi product quality from base-LM upgrade cadence. |

---

## 9. Code map — where to look for X

| You want to read about… | Open this file |
|---|---|
| Live REST + Gradio entrypoint (the deployed thing) | `app.py` |
| `POST /analysis` endpoint definition | `app.py:283-306` |
| LLM analysis prompts + OpenRouter call | `app.py:142-226` |
| TribeModel inference API | `tribev2/demo_utils.py` |
| Encoder architecture (the brain head) | `tribev2/model.py` |
| Feature extractor configs (what models, what layers) | `tribev2/grids/defaults.py:25-63` |
| Training experiment object | `tribev2/main.py` |
| PyTorch Lightning module (training loop) | `tribev2/pl_module.py` |
| The text → TTS → whisperx detour | `tribev2/demo_utils.py::TextToEvents` + `tribev2/eventstransforms.py::ExtractWordsFromAudio` |
| Surface projection / ROI utilities | `tribev2/utils_fmri.py` |
| Cortical rendering (PyVista / Nilearn) | `tribev2/plotting/cortical_pv.py`, `cortical.py` |
| Subcortical (volumetric) rendering | `tribev2/plotting/subcortical.py` |
| Dataset definitions | `tribev2/studies/{algonauts2025,lebel2023bold,lahner2024bold,wen2017}.py` |
| Training entry (local quick test) | `tribev2/grids/test_run.py` |
| Training entry (full Slurm grid) | `tribev2/grids/run_cortical.py`, `run_subcortical.py` |
| Live OpenAPI schema | `https://acti.cognition.ngrok.pro/openapi.json` |
| Public API contract (markdown) | `/home/ripper/ACTi_cognition/API.md` |

---

## 10. Environment & hardware

- **Host:** local box, 2× NVIDIA RTX PRO 6000 Blackwell Max-Q (97.9 GB VRAM each)
- **Important constraint:** TRIBE shares these GPUs with our vLLM DSV4 server on port 8000. **Only one can run at a time** — running both triggers CUDA OOM inside whisperx (reproducible).
- **Python:** 3.12 in `venv/`; project requires `>=3.11`
- **Key deps:** `torch>=2.5.1,<2.7`, `x_transformers==1.27.20`, `neuralset==0.0.2`, `neuraltrain==0.0.2`, `huggingface_hub`, `gtts`, `whisperx` (transitively)
- **Secrets:** `/home/ripper/ACTi_cognition/.env` — `HF_TOKEN`, `OPENROUTER_API_KEY`, `NGROK_AUTHTOKEN`, `NGROK_DOMAIN`
- **Startup:** `./run_tribe.sh` → activates venv → `python3 app.py` → loads model, mounts FastAPI + Gradio, spawns ngrok tunnel
- **Cache:** disk caches at `/home/ripper/ACTi_cognition/cache/` — feature extractor caches survive restarts and are content-keyed

---

## 11. Open questions worth asking the engineer to investigate

1. How much does the **TTS → ASR detour distort** word-timestamp distributions vs. true read-aloud audio? (i.e., does removing it actually hurt Pearson?)
2. Is the **audio-feature contribution meaningful for text inputs**, or is W2V-BERT essentially fitting noise on TTS audio? Ablation should be cheap.
3. Can the **LLM-stage be replaced** with a small task-specific model fine-tuned on (heatmap, stimulus) → analysis pairs? Once we have a few hundred examples, this is viable.
4. **Subject specificity vs. average-subject inference** — what's the Pearson gap on held-out stimuli? If significant, the API should accept an optional `subject_id`.
5. **Subcortical** prediction quality — the paper covers cortex; subcortical structures (amygdala, NAcc) are critical for the behavioral claims our LLM stage makes. What's the floor on those?

---

## 12. Things to share with the engineer alongside this brief

- The paper (you said you'll send it)
- HF page: `https://huggingface.co/facebook/tribev2`
- Repo source (this directory, or the upstream `github.com/facebookresearch/tribev2`)
- This file
- `API.md` (public-facing contract)
- A note that **LLM identity stays internal** — public artifacts must keep it abstract
