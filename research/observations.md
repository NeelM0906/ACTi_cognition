# Observations

## 2026-05-14

- Installed `pytest` into the project venv and verified the current targeted
  tests: 9 passed, 4 warnings.
- Direct text event construction was measured at about 0.0071 seconds versus
  9.7512 seconds for cold legacy TTS/ASR event construction on one short text.
  This supports a speed claim for event construction, not yet for fMRI quality.
- Direct text prediction agreed with cached legacy prediction at flat Pearson
  0.8616 on one short text. This is a proxy agreement result, not a benchmark
  fMRI result.
- On sub-01 Friends 6-chunk tiny-model screening, PearsonMSE improved held-out
  Pearson from 0.0152 to 0.0471 under the same split and cache.
- On sub-01 Friends 8-chunk tiny-model screening, PearsonMSE improved held-out
  Pearson from 0.0179 to 0.0461. This replicated the direction of the 6-chunk
  result.
- TemporalDelayBank with PearsonMSE produced 0.0477 on the 6-chunk run, only
  +0.0006 above PearsonMSE without delay. Treat as neutral.
- Retrieval top-1 was 1.0 in the tiny runs and is not useful for decisions at
  this scale.
- The 8-chunk MSE runtime includes cold feature-cache materialization, while
  the following PearsonMSE run was warm-cache. Do not use those timings to claim
  PearsonMSE is faster.
- Pre-registered next exploratory run:
  `algonauts_text_loss_episode_s01_e01_e08_val_e07_e08_3seed_v1`. It uses
  sub-01 Friends season 1 episodes e01-e08, text-only features, full episode
  holdout for e07/e08, seeds 33/34/35, tiny model, 10 epochs, and paired MSE vs
  PearsonMSE. This is exploratory, not confirmatory, because it is one subject
  and a tiny model.
- The first attempt at that run failed because the dataset loader validates
  Video events even for text-only ablations and the e05-e08 MKVs were not
  materialized. I fetched exactly the required files with DataLad.
- The second attempt failed because generated rows can have missing movie/chunk
  fields. I patched `AssignSplitByEpisode` to infer missing fields within a
  timeline and added tests for the behavior.
- The completed three-seed episode-held-out run gave MSE Pearson values
  0.0325, 0.0296, 0.0291 and PearsonMSE values 0.0580, 0.0603, 0.0613 for seeds
  33, 34, 35. Deltas were +0.0255, +0.0308, +0.0322; mean delta +0.0295, all
  positive. This is stronger exploratory evidence for PearsonMSE, still not a
  final benchmark claim.
- The next preregistered check is the same episode split and seeds without
  `--tiny-model`, to test whether the loss effect survives the default model
  scale before expanding the data split.
- The default-scale check did not support promoting high-weight PearsonMSE.
  Full-model MSE Pearson values were 0.0773, 0.0950, 0.0814; high-weight
  PearsonMSE values were 0.0790, 0.0757, 0.0609. Mean delta was -0.0127 and
  only one of three seeds was positive.
- Pre-registered diagnostic follow-up: run only two MSE-dominant correlation
  auxiliary settings on the same split/default model, `pearson_weight=0.1,
  mse_weight=1.0` and `pearson_weight=0.25, mse_weight=1.0`. This is an
  exploratory diagnostic based on the hypothesis that the high-weight Pearson
  objective discards useful amplitude information at full model capacity.
- The weak auxiliary setting `pearson_weight=0.1, mse_weight=1.0` had mean
  Pearson 0.0886 versus MSE mean 0.0846, but the deltas were +0.0143, +0.0020,
  -0.0042. Mixed seed sign means it is not promoted.
- The `pearson_weight=0.25, mse_weight=1.0` setting had mean Pearson 0.0813 and
  mean delta -0.0032. Reject this setting.
- Full-model MSE plus TemporalDelayBank lags=6,stride=2 had Pearson values
  0.0780, 0.0957, 0.0820 versus MSE 0.0773, 0.0950, 0.0814. All deltas were
  positive, but mean delta was only +0.00063, far below the meaningful gate.
  Treat as neutral.
- Pre-registered larger-split falsification run for only the weak auxiliary:
  sub-01 Friends s01 episodes e01-e12, hold out e11/e12, full model, seeds
  33/34/35, MSE vs `pearson_weight=0.1, mse_weight=1.0`. If this does not show
  positive deltas across seeds with a meaningful mean gain, stop the Pearson
  auxiliary line and move to a different modeling idea.
- The larger-split weak auxiliary falsification run completed. MSE Pearson
  values were 0.0805, 0.0870, 0.1124; weak auxiliary values were 0.0752,
  0.1022, 0.1104. Deltas were -0.0054, +0.0151, -0.0021; mean delta +0.0026
  with two negative seeds. This fails the predeclared gate. Stop the
  Pearson-loss line for now.
- Implemented an output-level TemporalDelayBank as a more faithful
  hemodynamic-lag test: the FIR sits after the subject/output head and before
  temporal pooling, with identity initialization. This tests per-output lag
  adaptation under MSE instead of continuing loss-weight tuning.
- Pre-registered next mechanistic check: same sub-01 Friends s01 e01-e12 split,
  hold out e11/e12, full model, seeds 33/34/35, MSE with output-level
  TemporalDelayBank lags=6,stride=2. Gate is at least +0.01 mean held-out
  Pearson and positive deltas across most seeds before any promotion.
- The output-level TemporalDelayBank run completed. Baseline MSE Pearson values
  from the same split were 0.0805, 0.0870, 0.1124. Output-delay MSE values were
  0.0688, 0.0921, 0.1146. Deltas were -0.0117, +0.0051, +0.0022; mean delta
  -0.0015. This fails the gate. Do not promote current delay-bank design.
- Next research decision: move to feature-quality rather than objective or
  delay tuning. Current ablations used the small `Qwen/Qwen3-0.6B` text
  extractor with one layer; the paper-scale direction is richer language
  features. The next controlled candidate should keep MSE and the split fixed
  while changing only text representation richness.
- Pre-registered feature-quality ablation: same sub-01 Friends s01 e01-e12
  split, hold out e11/e12, full model, seeds 33/34/35, MSE loss, same
  `Qwen/Qwen3-0.6B` text backbone but six fractional hidden layers
  `[0,0.2,0.4,0.6,0.8,1.0]` instead of one. Gate remains +0.01 mean Pearson
  and positive deltas across most seeds versus the same-split one-layer MSE
  baseline.
- Protocol note: this run logged `Extractor dims: {'text': (1, 1024)}`. The
  data pipeline mean-aggregated the six requested Qwen layers, so the completed
  ablation tests an averaged six-layer representation, not a concatenated
  multi-layer representation.
- Averaged six-layer Qwen result: baseline one-layer MSE Pearson values were
  0.0805, 0.0870, 0.1124; averaged six-layer values were 0.0812, 0.0725,
  0.0913. Deltas were +0.0007, -0.0145, -0.0211; mean delta -0.0116. Reject
  mean-aggregating these six layers.
- Next feature-quality step should first run a cheap dimension smoke check
  before a full ablation. Specifically, verify that a concatenated-layer or
  paper-scale text configuration actually changes `Extractor dims` as intended
  before paying the full cold-cache extraction cost.
- The concatenated-layer dimension smoke failed before training:
  `data.layer_aggregation` is validated as only `group_mean` or `mean`. This
  means concatenated hidden-layer text features are not currently reachable by
  that config switch. Logged as a tooling/protocol blocker, not a model result.
- A corrected dimension smoke with `data.layer_aggregation=None` succeeded and
  logged `Extractor dims: {'text': (6, 1024)}`. This is the valid path for a
  true six-layer Qwen feature ablation.
- Pre-registered true multi-layer feature ablation: same sub-01 Friends s01
  e01-e12 split, hold out e11/e12, full model, seeds 33/34/35, MSE loss,
  `Qwen/Qwen3-0.6B`, six layers `[0,0.2,0.4,0.6,0.8,1.0]`, and
  `data.layer_aggregation=None` to preserve the layer axis. Gate remains +0.01
  mean Pearson and positive deltas across most seeds versus same-split one-layer
  Qwen MSE.
- The true six-layer Qwen run completed with the intended
  `Extractor dims: {'text': (6, 1024)}`. Baseline one-layer MSE Pearson values
  were 0.0805, 0.0870, 0.1124; true six-layer values were 0.0908, 0.0834,
  0.0833. Deltas were +0.0103, -0.0037, -0.0291; mean delta -0.0075. Reject
  this representation.
- Current feature-quality decision: richer layers from the same small Qwen
  backbone are not helping. The next plausible feature test would need a
  genuinely stronger backbone or paper-scale text config, but it should be
  smoke-checked for shape/cache behavior first and treated as a separate
  hypothesis.
- Pre-registered next smoke: use the paper-style text backbone
  `meta-llama/Llama-3.2-3B` with six fractional layer boundaries
  `[0,0.2,0.4,0.6,0.8,1.0]` and `data.layer_aggregation=group_mean` on a
  two-chunk smoke split. Success criterion is only extractor/config viability
  and expected text feature shape; it is not a quality claim.
- The Llama-3.2-3B smoke succeeded. It logged
  `Extractor dims: {'text': (5, 3072)}` with peak VRAM about 13.5 GB at
  text-batch-size 1. This is a viable paper-style text-feature path.
- Pre-registered full paper-style text-backbone ablation: same sub-01 Friends
  s01 e01-e12 split, hold out e11/e12, full model, seeds 33/34/35, MSE loss,
  `meta-llama/Llama-3.2-3B`, six fractional layer boundaries with
  `group_mean` aggregation. Gate remains +0.01 mean Pearson and positive deltas
  across most seeds versus same-split Qwen one-layer MSE.
- Harness update: `experiments/algonauts_text_loss_ablation.py` now accepts
  `--features-to-use` plus per-modality cache/layer overrides so the next
  representation tests can add audio/video under the same split and seed policy
  instead of using one-off scripts.
- Text+audio smoke completed on the two-chunk tiny split while the full Llama
  text-feature run continued on GPU1. The run built 15 train and 15 validation
  audio chunks and logged `Extractor dims: {'text': (1, 1024), 'audio': (1,
  1024)}`. This only validates the multimodal path; the adjacent split,
  one-seed, one-batch Pearson value is not a quality signal.
- Current next-step policy from the specialist audit: finish the Llama full
  ablation first. If it passes the gate, freeze that text config and replicate
  on a second held-out episode split before broader claims. If it fails, run a
  paired text-only versus text+audio ablation as the next mechanistic branch,
  because adding acoustic/prosodic stimulus information is more scientifically
  grounded than continuing loss-weight or delay-bank tuning.
- The full paper-style Llama-3.2-3B group-mean text-feature run completed on
  the sub-01 Friends s01 e01-e12/e11-e12 split. Same-split Qwen MSE baseline
  Pearson values were 0.0805, 0.0870, 0.1124; Llama values were 0.1248, 0.1247,
  0.1266. Deltas were +0.0443, +0.0377, +0.0142; mean delta +0.0321. This
  passes the predeclared exploratory promotion gate.
- Interpretation: this is the first reliable model-quality improvement in this
  loop. The change is representation quality, not loss/benchmark tuning: it
  replaces the local small Qwen text features with the paper-style
  `meta-llama/Llama-3.2-3B` hidden-state representation under the same MSE
  encoder protocol. Correction: this is an improvement over the reduced Qwen
  proxy baseline only, not over the public leaderboard TRIBE baseline.
- Pre-registered split replication: freeze the Llama-3.2-3B group-mean config
  and rerun the same sub-01 Friends s01 e01-e12 pool with e09/e10 held out
  instead of e11/e12. Rerun a same-split Qwen MSE baseline and the frozen Llama
  candidate for seeds 33/34/35. This is a robustness replication within the
  available cached season-1 data, not a final test-set claim. Gate remains
  +0.01 mean held-out Pearson delta and positive deltas across most seeds.
- The frozen split replication passed. With e09/e10 held out, Qwen MSE Pearson
  values were 0.1076, 0.1160, 0.1162; frozen Llama values were 0.1689, 0.1601,
  0.1612. Deltas were +0.0613, +0.0441, +0.0450; mean delta +0.0502. This
  materially strengthens the Llama feature-quality finding, but it is still a
  validation-split result on one subject and one season subset.
- Decision after replication: do not continue searching over text backbones or
  losses on Friends s01 validation splits. The next scientific expansion should
  change the evidence axis: add real modalities (text+audio/text+video) or
  bring in more subjects/data, with the Llama text config frozen as the
  current text-only reference.
- Baseline correction after reviewing the leaderboard table and checkpoint:
  the public `facebook/tribev2` model already uses Llama text, Wav2Vec-BERT
  audio, V-JEPA video, fsaverage5 surface output, and multi-study training. Our
  local Qwen-vs-Llama runs were screening experiments on a 1000-parcel
  single-subject Algonauts subset. They do not establish a model stronger than
  the public baseline. Any future claim of improvement must compare against the
  public checkpoint or a faithful leaderboard-style retraining protocol.
- Challenge-data constraint: the exact public-leaderboard and OOD numbers in
  the paper cannot be reproduced from local labels alone. Friends season 7 and
  the OOD fMRI responses are withheld by the challenge and scored through
  Codabench. Local runs must be labeled as proxy/compatibility unless they use a
  fair held-out split drawn only from available training labels.
- Public-checkpoint compatibility smoke: `facebook/tribev2` text-only on
  sub-01 Friends `s01e01a` produced approximate surface-space mean Pearson
  0.1106 after projecting local Schaefer1000 labels to standard fsaverage5.
  This confirms the checkpoint/evaluator path, but it is not an official
  leaderboard metric and the chunk is in the public train distribution.
- Added a parcel-space option to `experiments/eval_pretrained_tribe_algonauts.py`
  to summarize fsaverage5 predictions back into projected Schaefer parcels. On
  the same text-only `s01e01a` smoke it produced approximate parcel mean Pearson
  0.1133 over 983 nonempty projected parcels. This is closer to the Algonauts
  scoring space than the surface proxy, but still depends on an approximate
  standard-fsaverage-to-subject-atlas mapping.
- Current active run: public `facebook/tribev2` full text+audio+video on the
  same `s01e01a` compatibility chunk. This is meant to calibrate the local
  evaluator and the multimodal feature path, not to make an improvement claim.
- Public full-multimodal compatibility result: on the same `s01e01a` chunk,
  approximate surface-space mean Pearson was 0.1604. In approximate
  parcel-space, the full multimodal run was 0.1667 before row matching.
- Matched text-only versus full-multimodal public TRIBE parcel comparison:
  because text-only and multimodal event grids kept different segment counts,
  both were rerun with array export and compared on nearest sample times within
  0.06 s. The matched local parcel mean Pearson was 0.1133 for text-only and
  0.1715 for text+audio+video, delta +0.0582 over 856 matched samples and 983
  projected parcels. This is a useful sanity check that the evaluator sees the
  expected multimodal benefit, not evidence that our algorithm beats TRIBE.
- Official submission blocker reduced: `target_sample_number` files for all
  four subjects are now materialized locally. They provide row counts for
  Friends S7 and OOD submissions but no hidden fMRI labels.
- Extended the public `facebook/tribev2` full-multimodal approximate parcel
  evaluator beyond `s01e01a`. On `s01e01b` the local mean Pearson was 0.1980
  over 983 projected parcels; on `s01e02a` it was 0.2133. These are still
  compatibility scores on available train-distribution labels, not official
  leaderboard or OOD scores.
- Tested a posthoc per-parcel temporal ridge calibration on exported public
  TRIBE predictions with fixed lags `[-4, -2, 0, 2, 4]` and alpha 10. On the
  within-`e01a` split it reduced mean Pearson from 0.1691 to 0.1512. On the
  stricter cross-chunk split, trained on `e01a/e01b` and validated on `e02a`,
  it reduced mean Pearson from 0.2106 to 0.1980. This branch is rejected.
- Decision: do not pursue shallow output calibration as the next improvement
  lever. The raw public TRIBE checkpoint is already well calibrated enough that
  local ridge fitting on limited labels overfits or removes useful signal. The
  next truthful ablation should change representation/training evidence under
  a fair held-out protocol rather than post-processing the benchmark target.
- Added a stronger spatiotemporal output-adapter test that lets the calibration
  use cross-parcel structure, still with fixed lags `[-4, -2, 0, 2, 4]` and a
  predeclared conservative ridge alpha of 1000. It also failed: on the
  within-`e01a` split mean Pearson fell from 0.1691 to 0.1056, and on
  train-`e01a/e01b` to val-`e02a` it fell from 0.2106 to 0.1899. This closes
  the shallow output-adapter line for now.
- Submission-format harness status: `experiments/predict_tribe_algonauts_submission.py`
  now emits Algonauts-style nested prediction dictionaries and zipped
  submission artifacts from `facebook/tribev2`. It uses local
  `target_sample_number` files for hidden-stimulus row counts, resamples model
  predictions to those sample times, projects fsaverage5 predictions into the
  1000-parcel target space, and records a manifest. This is a comparison
  infrastructure result, not a model-quality result.
- Friends S7 smoke: text-only public TRIBE for sub-01 `s07e01a` generated an
  all-finite `(460, 1000)` prediction file and zip. The final cached rerun took
  1.04 s and filled 7,820 nonfinite values caused by 17 empty
  fsaverage5-to-Schaefer projected parcels. No hidden labels were loaded or
  scored.
- Friends S7 full multimodal smoke: after materializing
  `friends_s07e01a.mkv` through the git-annex movie submodule, public TRIBE
  text+audio+video generated an all-finite `(460, 1000)` prediction file and
  zip for sub-01 `s07e01a`. It produced 685 prediction segments, filled the
  same 7,820 projected-parcel nonfinite values, and took 1,823.0 s. This is a
  real hidden-stimulus artifact suitable for official-format submission checks,
  but still has no local score because the Friends S7 fMRI labels are withheld.
- Operational implication: full multimodal hidden-stimulus inference is
  feasible but expensive, with V-JEPA/video extraction dominating runtime. To
  run the same ablation at leaderboard scope, the next gate is to materialize
  the remaining needed movies, cache raw public TRIBE predictions, and submit
  the raw baseline and candidate artifacts through the same official path. Any
  claim of improvement must come from that paired protocol or from a
  predeclared available-label validation split.
- Pre-registered local modality ablation after the leaderboard-table review:
  use public `facebook/tribev2` on sub-01 Friends `s01e02a`, approximate parcel
  metric, and the same evaluator for every condition. Compare text-only,
  audio-only, video-only, text+audio, text+video, audio+video, and the existing
  text+audio+video result. This will quantify the paper-style modality
  contribution on available local labels. It cannot reproduce the public
  leaderboard or OOD tables because those labels are withheld.
- The first local modality-ablation summary exposed an evaluator bug: when
  `audio` was requested without `video`, the evaluator removed the source Video
  event before `ExtractAudioFromVideo`, causing audio-only to run with no audio
  events and constant predictions. This `v1` summary is superseded.
- Corrected local modality ablation on `s01e02a`: approximate parcel mean
  Pearson is text 0.1356, audio 0.1530, video 0.1756, text+audio 0.1736,
  text+video 0.2108, audio+video 0.2029, and full text+audio+video 0.2133.
  Full multimodal remained best, but text+video was close behind at only
  -0.0025. This is a useful local sanity check that visual features dominate
  this particular chunk and audio contributes when extracted correctly; it is
  not evidence of a model stronger than the public TRIBE baseline.
- Tested scalar HRF target-offset selection as a mechanistic timing candidate.
  Offsets `[3, 4, 5, 6, 7]` seconds were scored on public TRIBE full-multimodal
  predictions for `e01a/e01b`; the default 5 s offset was best on train
  (`0.1817` mean Pearson), so validation on `e02a` was identical to baseline
  (`0.2133`, delta `0.0000`). This branch is rejected for now.
- Tested constrained late-fusion candidates using modality-specific public
  TRIBE outputs. A global convex blend over full/text+video/audio+video selected
  weights `0.30/0.35/0.35` for full/text+video/audio+video and improved
  `e02a` by only +0.0020 over full. A parcel-wise convex blend over the same
  three conditions improved by +0.0046 (`0.2179` vs `0.2133`) but remains below
  the +0.01 meaningful gate. A broader seven-condition parcel-wise fusion using
  0.5 s alignment tolerance was negative (-0.0011). Region-wise fusion is a
  plausible future direction, but current evidence is exploratory only and not
  a stronger-than-baseline result.
- Leave-one-chunk-out check for the clean parcel-wise fusion over `e01a`,
  `e01b`, and `e02a`: fold deltas versus full public TRIBE were +0.0035,
  +0.0067, and +0.0046. The direction is stable across all three folds, but the
  mean delta is only +0.00494, below the +0.01 gate. This strengthens the
  conclusion that region-wise fusion is real but currently too small to count
  as a meaningful improvement.
- Tested a more regularized group-wise fusion that shares one modality blend
  within Schaefer/Yeo networks. The 14 hemi-network mode had fold deltas
  +0.0007, +0.0046, and +0.0020, mean +0.00244. The 7-network mode had fold
  deltas +0.0007, +0.0043, and +0.0020, mean +0.00233. Both were positive on
  all three folds but weaker than parcel-wise fusion and below the +0.01 gate,
  so this does not support a meaningful improvement claim.
- Verification after adding group-wise fusion: focused pytest suite passed
  43/43 in 3.29 s. The remaining warnings are from existing neuralset,
  x-transformers, and Lightning behavior rather than this ablation.
- Local hidden-label audit for the leaderboard/OOD comparison: under
  `/home/ripper/data/tribe_benchmarks/download/algonauts_2025.competitors` I
  found hidden stimulus files such as `friends_s07e01a.wav`, but no local
  Friends S7/OOD fMRI response labels. Exact paper-table comparison therefore
  still requires official Codabench scoring or equivalent authorized labels.
- New subject-adaptation candidate: the public `facebook/tribev2` release
  checkpoint contains an average-subject output head. I extracted the frozen
  2048-dimensional low-rank TRIBE state before that head and fit a
  subject-specific ridge readout into the same Schaefer1000 parcel space. Alpha
  was selected only by inner folds inside the training chunks.
- Frozen-readout LOOCV over sub-01 Friends `s01e01a/e01b/e02a` cleared the
  local meaningful gate. Deltas versus the public full text+audio+video
  average head were +0.0329 on held-out `e01a`, +0.0371 on held-out `e01b`, and
  +0.0284 on held-out `e02a`; mean delta +0.03282. This is the first local
  candidate that is meaningfully stronger than the real public TRIBE baseline
  under our available-label protocol.
- The frozen-readout result should not yet be called a leaderboard win. It is
  still a local available-label compatibility result on train-distribution
  Friends chunks. The next required step is to scale the same predeclared
  subject-specific readout across more chunks/subjects and produce official
  hidden Friends S7/OOD submissions for Codabench scoring.
- Verification after frozen-readout addition: full focused research suite
  passed 47/47 in 3.36 s. Remaining warnings are existing neuralset,
  x-transformers, and Lightning warnings.
- Matched supervised calibration control: a ridge readout from the public
  average-head parcel predictions to the subject's parcels also cleared the
  local gate. Fold deltas were +0.0248, +0.0394, and +0.0210; mean +0.02839.
  This means the robust discovery is supervised subject adaptation over the
  public TRIBE output, not yet a decisive advantage for low-rank features.
- Low-rank readout versus matched prediction-ridge control: low-rank is ahead
  on mean by +0.00443, but fold differences are mixed (+0.0081, -0.0023,
  +0.0075). Treat low-rank as promising, not proven superior to simpler
  supervised calibration, until tested on more chunks and subjects.
- Review downgrade: the current subject-adaptation evidence is chunk-held-out
  over `e01a/e01b/e02a`, not episode-held-out. It also uses the local
  1 Hz TRIBE segment grid with interpolated fMRI targets, not native fMRI TR
  rows or official leaderboard scoring. The result remains useful, but the next
  confirmatory run must hold out complete episodes and enforce row-alignment
  guards.
- Added guardrails to the readout harness after review: train/validation run
  overlap is rejected, and cached low-rank feature segment starts must match
  baseline sample times after the fixed 5 s target offset. The existing three
  cached readout feature files pass this alignment check.
- Verification after guard changes: full focused research suite passed 49/49 in
  3.30 s. Remaining warnings are existing dependency/lightning warnings.
- The readout and prediction-ridge harnesses now enforce episode-level split
  disjointness. A train/validation split such as `e01a` vs `e01b` is rejected
  because those are halves of the same episode. The scripts also validate
  baseline sidecar metadata for checkpoint, features, metric space, target
  offset, subject/movie/chunk, row count, and shapes before scoring.
- Completed the missing public full TRIBE baseline for sub-01 Friends
  `s01e02b`: approximate projected-parcel mean Pearson was `0.2239`, with
  712 finite target rows and 983 nonempty projected parcels. Runtime was
  dominated by full multimodal video extraction (`predict_seconds` 2254.6 s).
- Episode-heldout subject-adaptation check over sub-01 Friends `s01e01` and
  `s01e02`: frozen low-rank readout improved over the raw public average head
  in both directions. Train `e01a/e01b`, validate `e02a/e02b`: `0.2590` vs
  `0.2185`, delta `+0.0405`. Train `e02a/e02b`, validate `e01a/e01b`:
  `0.2235` vs `0.1817`, delta `+0.0418`. Mean delta `+0.04112`.
- Matched episode-heldout prediction-ridge calibration also improved over the
  raw public average head in both directions. Train `e01`, validate `e02`:
  `0.2552` vs `0.2185`, delta `+0.0367`. Train `e02`, validate `e01`:
  `0.2239` vs `0.1817`, delta `+0.0422`. Mean delta `+0.03943`.
- Truthful interpretation of the episode-heldout result: supervised
  subject-adaptation is now supported under a stronger local episode-disjoint
  protocol. Low-rank readout still is not established as a distinct model-level
  improvement over the simpler matched prediction-ridge control: low-rank was
  only `+0.00170` ahead on mean, winning one direction by `+0.00380` and losing
  the reverse by `-0.00041`. The result remains a 1 Hz interpolated-label local
  proxy and is not directly comparable to the Algonauts public or OOD tables.
- Verification after episode-heldout run and guard updates: focused research
  suite passed 51/51 in 3.34 s, and TSV schema validation found no malformed
  rows in `results.tsv`, `research/scientific_ledger.tsv`, or
  `research/artifacts_manifest.tsv`.
- Added a native-fMRI-TR subject-adaptation evaluator. It resamples public
  TRIBE projected-parcel predictions and frozen low-rank features to the
  recorded Algonauts TR grid (`1/1.49 Hz`) before fitting/scoring ridge
  readouts. This removes the previous 1 Hz interpolated-target proxy as the
  primary local check, while still remaining an available-label local proxy.
- Native-TR episode-heldout result over sub-01 Friends `s01e01` and `s01e02`:
  train `e01a/e01b`, validate `e02a/e02b` gave raw public head `0.2047`,
  prediction-ridge `0.2384` (`+0.0337`), and low-rank readout `0.2406`
  (`+0.0359`). Train `e02a/e02b`, validate `e01a/e01b` gave raw public head
  `0.1687`, prediction-ridge `0.2090` (`+0.0403`), and low-rank readout
  `0.2078` (`+0.0391`).
- Native-TR interpretation: the subject-adaptation signal survives the stricter
  native-row check. Mean deltas versus raw public average head were `+0.036996`
  for prediction-ridge and `+0.037522` for low-rank. Low-rank remains
  unsupported as a separate model-level improvement because its mean advantage
  over prediction-ridge was only `+0.000525`, winning one direction and losing
  the reverse.
- Verification after native-TR evaluator addition: focused research suite
  passed 53/53 in 3.32 s. TSV schema validation again found no malformed rows
  in `results.tsv`, `research/scientific_ledger.tsv`, or
  `research/artifacts_manifest.tsv`.
- Scaled the native-TR episode-heldout check to sub-02 after materializing its
  Friends fMRI and atlas files through git-annex. Public full TRIBE 1 Hz local
  baseline means for sub-02 were `0.1807` (`e01a`), `0.2046` (`e01b`),
  `0.2150` (`e02a`), and `0.2156` (`e02b`).
- Sub-02 native-TR episode-heldout result: train `e01a/e01b`, validate
  `e02a/e02b` gave raw public head `0.1960`, prediction-ridge `0.2379`
  (`+0.0418`), and low-rank `0.2407` (`+0.0446`). Train `e02a/e02b`,
  validate `e01a/e01b` gave raw public head `0.1756`, prediction-ridge
  `0.2165` (`+0.0409`), and low-rank `0.2127` (`+0.0371`).
- Two-subject native-TR aggregate over sub-01 and sub-02: prediction-ridge
  calibration improved over the raw public average head by mean delta
  `+0.039185`; frozen low-rank readout improved by `+0.039193`. Low-rank minus
  prediction-ridge was `+0.000007`, effectively zero. This strengthens the
  claim that subject-specific supervised adaptation is real, and weakens any
  claim that low-rank is the better algorithmic choice.
- Recovery note after lost connection: no research process was still running.
  The only GPU compute process was `python3 app.py` for the ACTi/TRIBE app on
  GPU 0. The last incomplete branch was the planned sub-03/sub-05 scale-out of
  the native-TR subject-adaptation protocol.
- Recovered prior sub-03/sub-05 materialization: `git-annex` had completed the
  sub-03 and sub-05 Friends fMRI h5 plus Schaefer atlas files. The prior run
  had also completed sub-03 public full TRIBE approximate parcel baseline arrays
  for `s01e01a/e01b/e02a/e02b`, with local 1 Hz means `0.2074`, `0.2189`,
  `0.2200`, and `0.2233`.
- Completed the missing sub-05 public full TRIBE baseline arrays. Local 1 Hz
  approximate parcel means were `0.1459` (`e01a`), `0.1880` (`e01b`), `0.2068`
  (`e02a`), and `0.1867` (`e02b`). These are still baseline materialization
  artifacts, not leaderboard scores.
- Sub-03 native-TR episode-heldout result: train `e01a/e01b`, validate
  `e02a/e02b` gave raw public head `0.2090`, prediction-ridge `0.2665`
  (`+0.0576`), and low-rank readout `0.2676` (`+0.0587`). Train `e02a/e02b`,
  validate `e01a/e01b` gave raw public head `0.1974`, prediction-ridge
  `0.2447` (`+0.0474`), and low-rank `0.2437` (`+0.0463`).
- Sub-05 native-TR episode-heldout result: train `e01a/e01b`, validate
  `e02a/e02b` gave raw public head `0.1877`, prediction-ridge `0.2196`
  (`+0.0318`), and low-rank `0.2176` (`+0.0299`). Train `e02a/e02b`,
  validate `e01a/e01b` gave raw public head `0.1559`, prediction-ridge
  `0.1825` (`+0.0266`), and low-rank `0.1847` (`+0.0288`).
- Four-subject native-TR aggregate over sub-01/sub-02/sub-03/sub-05:
  prediction-ridge calibration improved over the raw public average head by
  mean delta `+0.040016`; frozen low-rank readout improved by `+0.040057`.
  All four subjects were positive versus the raw head. Low-rank minus
  prediction-ridge remained effectively zero at `+0.000041`, so low-rank still
  does not clear the model-level gate over the simpler calibration control.
- Verification after recovery continuation: TSV schema checks passed for
  `results.tsv`, `research/scientific_ledger.tsv`, and
  `research/artifacts_manifest.tsv`. The full focused research suite passed
  `53/53` in 3.34 s; remaining warnings are existing neuralset, Lightning, and
  x-transformers warnings.
- Next iteration goal set: produce paired official-format hidden-stimulus
  artifacts for the raw public TRIBE average head and the subject-specific
  prediction-ridge calibrated candidate. The first gate is a shape/finite/no
  hidden-label smoke on existing `sub-01/s07e01a`; the full gate remains
  Codabench scoring against withheld Friends S7/OOD labels.
- Added `experiments/calibrate_tribe_submission.py`. It loads a raw
  submission-shaped prediction dictionary, fits a per-subject ridge calibration
  map on available Friends season-1 labeled baseline arrays, selects alpha by
  complete-episode inner folds, applies the map to hidden-stimulus predictions,
  and writes `.npy`, optional `.zip`, fitted calibrator `.npz`, and a manifest.
  The implementation is vectorized and uses BLAS-backed ridge solves; applying
  a fitted map to the existing smoke file took ~0.03 s for the item and 3.31 s
  end-to-end including fitting.
- Calibrated submission smoke completed for `sub-01/s07e01a` using the existing
  raw full multimodal hidden prediction file. The output shape was `(460,1000)`,
  all values were finite, 983 parcels were calibrated, and 17 uncalibrated
  units were preserved from the raw prediction. The selected alpha was `100`.
  This is a submission-mechanics smoke only, not an accuracy result.
- Verification for the calibrated submission utility: focused tests passed
  `8/8` in 3.24 s, TSV schema checks passed for all autoresearch ledgers, and
  the full focused research suite passed `55/55` in 3.33 s. Remaining warnings
  are existing neuralset, Lightning, and x-transformers warnings.
- H001-S harness work: added `experiments/tribe_faithful_retrain.py` and
  `tests/test_tribe_faithful_retrain.py`, then fixed a split-transform bug
  where `ConfDict.update` merged `AssignSplitByEpisode` with the default
  `SplitEvents` `val_ratio`. Focused tests over the new harness plus event
  split/config tests passed `11/11`.
- H001-S smoke attempt is blocked, not passed. The fixed harness built the
  intended complete-episode split (`s01e01` train, `s01e02` validation) and
  completed cold text and audio feature extraction under
  `h001_faithful_retrain_v1`, but cold video extraction projected roughly
  54 windows x 132 s per window before any training. I stopped the run at
  1518 s to avoid a certain budget breach. No `(n_TRs, 1000)` model output was
  produced, so the gate is not cleared. Next action: downgrade the H001-S
  protocol to reuse an audited feature cache or split feature-cache warmup from
  the retrain smoke before rerunning.
- H001-S text-only protocol downgrade passed as a smoke, not a quality claim.
  After fixing a result serialization bug in `a9ead79` (`ConfDict.to_dict` is
  not available in this environment), the run trained the tiny model on
  `sub-01` `s01e01`, evaluated on held-out `s01e02`, and produced an all-finite
  validation prediction array with shape `(200,1000)`. Runtime was 7.88 s,
  `test/pearson` was `0.0038157`, and focused H001-S/event/config tests passed
  `12/12`. This clears only the downgraded text-only training-pipeline gate;
  the full multimodal path remains pending behind `H000-VCACHE-SM` and
  `H001-SM`.
