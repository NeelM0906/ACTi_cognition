# ACTi TRIBE Autoresearch Program

This project adapts the `karpathy/autoresearch` operating model to TRIBE/ACTi
brain-encoding research. The goal is not to chase a public leaderboard by
tuning to held-out labels. The goal is to improve the model and algorithm using
scientifically defensible changes, controlled ablations, and complete logging.

## Research Objective

Improve TRIBE/ACTi along two axes:

1. Benchmark-relevant brain prediction quality, primarily Pearson correlation
   on held-out fMRI responses.
2. Runtime and reliability, especially for the text-input path used by ACTi.

## Hard Guardrail

Never reward-hack or benchmark-optimize. Do not tune to the final holdout, leak
episode/chunk neighbors across splits, hide failed runs, compare cache-warm
against cache-cold runtime, or claim progress from qualitative LLM-stage appeal.
Science and technologically backed results only.

## In-Scope Files

Read these before starting or resuming a research loop:

- `RESEARCH_ENGINEER_BRIEF.md`: product, pipeline, speed and quality levers.
- `README.md`: repository context.
- `tribev2/model.py`: brain encoder architecture.
- `tribev2/pl_module.py`: training loss and metrics.
- `tribev2/eventstransforms.py`: event processing and split assignment.
- `tribev2/studies/algonauts2025.py`: dataset timelines and fMRI loading.
- `experiments/`: ablation harnesses.
- `research/`: methodology, guardrails, ledger, claims, observations.

## Required Logs

Maintain these files throughout the work:

- `results.tsv`: compact autoresearch-style run ledger.
- `research/scientific_ledger.tsv`: detailed scientific ledger.
- `research/observations.md`: interpretation notes, anomalies, failures.
- `research/artifacts_manifest.tsv`: JSONs, logs, cache versions, provenance.
- `research/methodology.md`: current protocol and promotion gates.
- `research/benchmark_guardrails.md`: anti-hacking rules.
- `research/claims.md`: claims currently supported by evidence.

## Experiment Loop

1. State the hypothesis before running.
2. Record the split policy, cache versions, seeds, command, and artifact paths.
3. Run the baseline and candidate under the same protocol.
4. Log all runs, including crashes and negative results.
5. Promote only if the improvement survives the predeclared gate.
6. If a result is cache-confounded, split-confounded, one-seed, or tiny-model
   only, label it smoke or exploratory.

## Current Direction

PearsonMSE and the current delay-bank designs are no longer active quality
branches: they helped weakly or locally but failed the default-scale
larger-split gates. The Qwen-to-Llama text-feature result is now treated only
as a reduced-proxy screening result because the public TRIBE baseline already
uses Llama, audio, video, and multi-study training.

The active direction is now paired official-format submission generation. Local
native-TR proxy evidence supports subject-specific prediction-ridge calibration
over the raw public `facebook/tribev2` average head, but exact public/OOD
leaderboard comparison still requires Codabench because Friends season 7 and
OOD labels are withheld. The next iteration must produce raw-public and
prediction-ridge-calibrated hidden-stimulus artifacts under the same submission
format, with manifests proving shape, finiteness, cache/provenance, alpha
selection, and no hidden-label access.
