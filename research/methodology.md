# Research Methodology

## Evidence Levels

- Smoke: verifies that code runs and that a direction is plausible. Tiny subsets,
  adjacent chunks, one seed, or `limit_train_batches` belong here.
- Exploratory: fixed train/validation policy, baseline and candidate rerun under
  the same protocol, but still small scale or one subject.
- Confirmatory: predeclared split, repeated seeds, no tuning on final holdout,
  paired statistics, and baseline rerun under the same commit/environment.

## Primary Metric

The primary quality metric is held-out Pearson correlation between predicted and
target fMRI responses. Retrieval top-1 is not decision-grade on tiny subsets
because it saturated at 1.0 in initial runs.

## Split Policy

For Friends data, hold out complete episodes, not neighboring chunks from an
episode. A candidate must not be tuned using the final holdout. Validation is
for model-selection decisions; final test is for claims.

Current next protocol:

- Subjects: `Algonauts2025/sub-01`, `sub-02`, `sub-03`, and `sub-05`.
- Task: Friends, season 1.
- Baseline: public `facebook/tribev2` full text+audio+video average-subject
  head in the same local projected-parcel metric space.
- Candidate: supervised subject adaptation. Compare both a simple
  prediction-ridge calibration over the public parcel predictions and a frozen
  low-rank TRIBE readout. The low-rank method is only a model-level lead if it
  beats the prediction-ridge control, not merely the raw public average head.
- Split: complete-episode heldout. Current local proxy uses `s01e01` halves
  versus `s01e02` halves in both directions; adjacent halves of the same episode
  are not valid train/validation splits for claims.
- Promotion gate versus raw public head: candidate must beat the same-protocol
  raw average head by at least +0.01 absolute Pearson on held-out episodes.
- Promotion gate for low-rank-specific claims: frozen low-rank readout must beat
  matched prediction-ridge calibration by at least +0.01 mean Pearson with
  positive deltas across most held-out episodes/subjects.
- Current official-submission candidate: subject-specific prediction-ridge
  calibration over public TRIBE parcel predictions. Fit calibration maps using
  only available Friends season-1 labels, select alpha by complete-episode
  inner folds, then apply those maps to raw hidden Friends S7/OOD prediction
  files. Hidden labels must not be loaded or inferred.
- Current submission iteration goal: produce paired official-format artifacts
  for raw public TRIBE and prediction-ridge-calibrated predictions. First smoke
  target is `sub-01/s07e01a`; scale to all materialized Friends S7/OOD stimuli
  only after raw/candidate manifests verify shape, finiteness, provenance, and
  no hidden-label access.
- Stopped branch: PearsonMSE/loss-weight tuning is paused because the
  default-scale larger-split falsification run was mixed and below the gate.
- Stopped branch: current hidden and output TemporalDelayBank variants are
  paused because the gains were below threshold or negative on episode-held-out
  default-scale checks.
- Before launching cold feature-cache ablations, run a cheap dimension smoke
  check and verify that `Extractor dims` matches the intended representation
  shape. If it does not, log the protocol mismatch and do not interpret the run
  as the intended feature test.
- For multimodal follow-ups, first run a smoke check that validates event
  transforms, modality chunk counts, extractor dimensions, and cache versions.
  A one-batch adjacent-chunk smoke may only support viability, never quality.
- Because the Llama text-only improvement has now passed two season-1
  validation splits, do not keep tuning text/loss variants on Friends s01. Any
  next quality claim must add an evidence axis: a new modality, new subject(s),
  or a stricter held-out data slice.

## Runtime Policy

Runtime claims must compare the same cache state: cold vs cold, or warm vs warm.
Feature extraction, training, and evaluation time should be separated where
possible. Cache-confounded timings can be logged but not used as speed claims.

## Promotion Rules

Promote a method only when:

- The baseline was rerun under the same code and data protocol.
- The split does not leak adjacent chunks from the same episode.
- Seeds and failed runs are logged.
- The result is not explained by cache state or data availability.
- The added complexity is justified by a meaningful metric gain.
- "Stronger than baseline" means stronger than the public leaderboard-style
  TRIBE baseline (`facebook/tribev2` or a faithful retrain), not stronger than
  the reduced Qwen proxy used for cheap local screening.
- Exact reproduction of the public leaderboard and OOD tables requires official
  Codabench scoring because Friends season 7 and OOD fMRI labels are withheld.
  Locally scored public-checkpoint runs must be marked as approximate
  compatibility/proxy evidence unless they are performed on a predeclared
  held-out split from available training labels.
- When comparing public fsaverage5 checkpoints against parcel-trained local
  models, prefer a common metric space. The current evaluator supports two
  approximate bridges: project Schaefer targets to fsaverage5, or summarize
  fsaverage5 predictions into projected Schaefer parcels. Neither is a
  substitute for official preprocessing/scoring.
