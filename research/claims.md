# Claims

## Supported

- Direct text event construction removes the text-input TTS/ASR detour and is
  much faster for event construction on the measured short stimulus.
- PearsonMSE showed a real small-scale exploratory signal only in tiny
  sub-01 Friends text-only runs. The strongest tiny-model result is a
  three-seed episode-held-out run with mean absolute Pearson delta +0.0295 over
  MSE and all seed deltas positive.
- TemporalDelayBank is implemented and tested, but current evidence does not
  show a meaningful gain.
- Output-level TemporalDelayBank is implemented and unit-tested as a more
  faithful per-output lag mechanism, but current benchmark evidence is
  negative.
- The text+audio ablation path is operationally viable on a tiny smoke run:
  it builds audio chunks and exposes text/audio extractor dimensions. This is
  not evidence of a quality improvement.
- Paper-style `meta-llama/Llama-3.2-3B` group-mean text features are supported
  as an exploratory improvement over our local reduced small-Qwen proxy
  baseline on sub-01 Friends season-1 validation splits. It passed two frozen
  episode-held-out checks: +0.0321 mean Pearson on e11/e12 and +0.0502 on
  e09/e10, each across three seeds with all seed deltas positive.
- The public `facebook/tribev2` checkpoint can now be evaluated locally in two
  approximate compatibility modes: Schaefer1000 target labels projected to
  fsaverage5 surface, and fsaverage5 predictions summarized back into projected
  Schaefer1000 parcels. These are evaluator capabilities, not model-quality
  improvements.
- In the local approximate parcel evaluator, public `facebook/tribev2`
  text+audio+video beats the same checkpoint run text-only on a single
  train-distribution Friends chunk by +0.0582 matched mean Pearson. This
  supports the evaluator and expected modality contribution, but it is not a
  claim about a new model.
- The local public-checkpoint evaluator now has array-export coverage on three
  full-multimodal train-distribution chunks (`s01e01a`, `s01e01b`, `s01e02a`),
  which is sufficient for small fair local ablations against raw public TRIBE
  predictions on available labels.
- The official-format submission harness is operational for a one-split
  hidden Friends S7 smoke: it generated all-finite nested `.npy` and `.zip`
  artifacts for `facebook/tribev2` text-only and full text+audio+video on
  sub-01 `s07e01a`, with manifests documenting row counts, fill counts, cache
  use, and runtime.
- On one available-label local compatibility chunk (`sub-01` Friends
  `s01e02a`), the corrected public TRIBE modality ablation follows the expected
  pattern: full text+audio+video was best at 0.2133 approximate parcel mean
  Pearson, text+video was nearly tied at 0.2108, audio+video was 0.2029,
  video-only was 0.1756, text+audio was 0.1736, audio-only was 0.1530, and
  text-only was 0.1356. This supports the evaluator and paper-style modality
  ablation direction locally.
- A frozen public-TRIBE encoder with a subject-specific ridge readout is
  supported as a meaningful local improvement over the public full
  average-subject head on the current chunk-held-out, 1 Hz interpolated-label
  protocol. Across
  leave-one-chunk-out folds over `s01e01a/e01b/e02a`, mean delta was +0.03282
  with all three folds positive (+0.0329, +0.0371, +0.0284), clearing the +0.01
  gate. This is not yet an official hidden leaderboard claim.
- A matched supervised ridge calibration over public average-head parcel
  predictions is also supported as a meaningful local improvement over the
  public full average-subject head. It reached mean delta +0.02839 with all
  three folds positive. Therefore the strongest current claim is supervised
  subject adaptation over public TRIBE, not a fully unsupervised or zero-shot
  model improvement.
- Under a stricter local episode-heldout protocol over sub-01 Friends
  `s01e01` and `s01e02`, supervised subject adaptation remains supported.
  Frozen low-rank readout improved over the raw public full average-subject
  head by mean delta +0.04112 across the two episode directions, and matched
  prediction-ridge calibration improved by mean delta +0.03943. This is still a
  local 1 Hz projected-parcel proxy, not a hidden leaderboard result.
- Under native fMRI-TR scoring over the same sub-01 Friends `s01e01`/`s01e02`
  episode-heldout protocol, supervised subject adaptation remains supported.
  Prediction-ridge calibration improved over the raw public full average head
  by mean delta +0.036996, and frozen low-rank readout improved by mean delta
  +0.037522. This is the strongest local evidence so far because it scores on
  recorded fMRI rows rather than interpolating targets to TRIBE's 1 Hz segment
  grid.
- The native-TR subject-adaptation signal replicated on sub-02 and across a
  two-subject aggregate. Across sub-01 and sub-02, prediction-ridge calibration
  improved over the raw public full average head by mean delta +0.039185, and
  frozen low-rank readout improved by mean delta +0.039193.
- The native-TR subject-adaptation signal also replicated after scale-out to
  sub-03 and sub-05. Across sub-01, sub-02, sub-03, and sub-05,
  prediction-ridge calibration improved over the raw public full average head
  by mean delta +0.040016, and frozen low-rank readout improved by mean delta
  +0.040057. All four subjects were positive versus the raw public head.
- The official-format calibrated submission path is now mechanically supported
  for at least the existing `sub-01/s07e01a` smoke artifact. It fits the
  prediction-ridge map using only available season-1 labels and applies it to a
  hidden-stimulus prediction file, producing finite `(460,1000)` output with
  manifest provenance. This supports pipeline viability only, not accuracy.

## Not Yet Supported

- High-weight PearsonMSE is not a default-model improvement on the current
  single-subject episode-held-out split. It underperformed MSE on mean Pearson
  at default scale, so it should not be promoted as-is.
- Weak Pearson auxiliary loss is not supported as a default-scale improvement.
  The larger e01-e12 falsification run had mean delta +0.0026 with two negative
  seeds, below the gate.
- TemporalDelayBank should not be presented as an improvement yet.
- MSE plus TemporalDelayBank has only a very small all-positive delta on the
  current default-scale split. It is below the meaningful threshold.
- MSE plus output-level TemporalDelayBank is not supported. On the larger
  e01-e12 split it had mean delta -0.0015 versus same-split MSE.
- Averaged six-layer `Qwen/Qwen3-0.6B` text features are not supported. They
  underperformed the one-layer same-backbone baseline by -0.0116 mean Pearson
  on the larger e01-e12 split.
- True six-layer `Qwen/Qwen3-0.6B` text features with preserved layer axis are
  not supported. They underperformed the one-layer same-backbone baseline by
  -0.0075 mean Pearson on the larger e01-e12 split.
- Training runtime is not improved by PearsonMSE based on current artifacts;
  existing timing is cache-confounded.
- Direct text path quality is not yet confirmed against fMRI; current
  direct-vs-legacy prediction agreement is only a proxy.
- Text+audio multimodal encoding is not yet supported as a model improvement.
  The current evidence is only a one-batch adjacent-chunk smoke check.
- The current evidence does not show an improvement over the public
  leaderboard TRIBE baseline. That baseline is the `facebook/tribev2`
  leaderboard-style checkpoint with Llama text, Wav2Vec-BERT audio, V-JEPA
  video, and multi-study training. Our positive Llama runs were trained from
  scratch on a small local 1000-parcel Algonauts slice against a Qwen proxy.
- The local `s01e01a` public-checkpoint scores are not official Algonauts
  scores. Friends season 7 and OOD ground-truth labels are withheld, so exact
  comparison to the paper's public/OOD leaderboard tables requires an official
  Codabench submission or access to the hidden labels.
- The local multimodal-over-text result should not be interpreted as a
  publishable ablation. It used a chunk likely included in public TRIBE's
  training distribution and exists only to validate the evaluation machinery.
- Posthoc per-parcel temporal ridge calibration is not supported. It lost to
  raw public TRIBE on both a within-chunk split (`0.1512` vs `0.1691`, delta
  `-0.0179`) and a cross-chunk split trained on `e01a/e01b` and validated on
  `e02a` (`0.1980` vs `0.2106`, delta `-0.0126`).
- A global spatiotemporal ridge output adapter is also not supported. With
  fixed alpha `1000`, it lost within chunk (`0.1056` vs `0.1691`, delta
  `-0.0635`) and cross chunk (`0.1899` vs `0.2106`, delta `-0.0207`).
- Scalar HRF target-offset selection is not supported as an improvement. Among
  `[3, 4, 5, 6, 7]` seconds, the default 5 s offset was selected on `e01a/e01b`
  and tied baseline exactly on held-out `e02a` (`0.2133`, delta `0.0000`).
- Global and parcel-wise late fusion are not yet supported as meaningful
  improvements. The clean global fusion over full/text+video/audio+video was
  +0.0020 on held-out `e02a`; the clean parcel-wise fusion was +0.0046; both
  are below the +0.01 gate. The broader seven-condition parcel-wise fusion was
  negative (-0.0011) under a looser 0.5 s alignment tolerance.
- The leave-one-chunk-out follow-up makes the parcel-wise fusion signal more
  credible but still not meaningful: all three folds were positive
  (`+0.0035`, `+0.0067`, `+0.0046`), mean `+0.00494`, which remains below the
  `+0.01` promotion gate.
- Group-wise fusion is also not supported as a meaningful improvement. Sharing
  one blend within 14 hemi-networks reached mean delta `+0.00244`, and sharing
  within 7 networks reached `+0.00233`; both were below the `+0.01` gate and
  weaker than parcel-wise fusion.
- Low-rank readout superiority over the matched supervised prediction-ridge
  control is not yet supported. In the earlier chunk-heldout check, low-rank was
  ahead by only +0.00443 mean delta and lost one of three folds. In the stricter
  1 Hz episode-heldout check, low-rank was ahead by only +0.00170 mean delta
  and won one of two episode directions. In the native-TR episode-heldout check,
  low-rank was ahead by only +0.000525 mean delta and again won only one of two
  episode directions for sub-01. After adding sub-02, the two-subject native-TR
  low-rank advantage was only +0.000007, effectively zero. After adding sub-03
  and sub-05, the four-subject native-TR low-rank advantage is still only
  +0.000041, far below the +0.01 model-level gate.
- Leaderboard-faithful subject-adaptation claims are not yet supported. The
  best current subject-adaptation evidence is now episode-heldout and native-TR
  locally, but it still uses available Friends season-1 labels rather than
  official hidden Friends S7/OOD scoring.
- The new hidden Friends S7 submission artifacts are not accuracy results.
  They use hidden-stimulus sample counts only; the fMRI labels remain withheld,
  so they cannot be compared to the paper leaderboard table without Codabench
  scoring or equivalent authorized labels.
- The calibrated `sub-01/s07e01a` Friends S7 artifact is also not an accuracy
  result. It proves the candidate submission format and calibration mechanics,
  but no hidden fMRI labels were loaded and no Codabench score has been
  obtained.
- The project still does not have official hidden-leaderboard evidence of a
  model that is meaningfully stronger than the public TRIBE leaderboard
  baseline. The latest accepted progress is local subject-adaptation evidence
  plus a fair route to generate and compare raw public baseline and candidate
  submissions under the same protocol.
- The local `s01e02a` modality ablation should not be generalized to the
  leaderboard table or OOD movie table. It is one subject, one train-distribution
  chunk, approximate parcel projection, and a public checkpoint that may have
  seen the same distribution. The first `v1` modality summary is superseded by
  `v2` because the evaluator initially pruned source video before audio
  extraction for audio-without-video conditions.

## Current Decision

Stop the Pearson-loss, delay-bank, same-backbone Qwen layer-richness, shallow
temporal-ridge, and global output-adapter lines. Use `facebook/tribev2` / the
leaderboard-style average head as the baseline for future "stronger than
baseline" claims. The Qwen proxy remains useful only for cheap local screening
and cannot establish progress over the real baseline.

Current lead: supervised subject adaptation over public TRIBE clears the local
native-TR episode-heldout gate across four subjects, with prediction-ridge
calibration at +0.040016 and frozen low-rank readout at +0.040057 versus the
public full average head. Because the matched prediction-ridge control explains
essentially all of the gain and is simpler, treat subject-specific
prediction-ridge calibration as the current practical candidate for
official-format submissions. Do not claim a leaderboard improvement until the
raw public head and calibrated candidate are scored through Codabench or an
equivalent authorized hidden-label path.
