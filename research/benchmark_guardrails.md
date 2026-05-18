# Benchmark Guardrails

These rules are mandatory for ACTi/TRIBE research.

1. Do not tune on the final holdout.
2. Do not use Friends season 7 labels as a development signal.
3. Do not split neighboring chunks from the same episode across train and validation
   when making quality claims.
4. Do not compare candidate and baseline runs with different cache states for
   speed claims.
5. Do not report only successful runs. Log crashes, OOMs, failed hypotheses, and
   neutral results.
6. Do not promote a method based on retrieval top-1 when it is saturated.
7. Do not use the ACTi vision-LLM qualitative output as the reward signal for
   brain-model improvements.
8. Do not claim a benchmark improvement from one subject, one movie subset, one
   seed, tiny model, or partial training. Label those results smoke or exploratory.
9. Do not add benchmark-specific branches that exploit dataset names, chunk ids,
   transcript quirks, or label availability rather than improving the model.
10. Prefer changes with a plausible neuroscience or representation-learning
    rationale, then test them under paired ablations.
11. Treat Friends season 7 and OOD labels as unavailable final holdouts unless
    obtained through official challenge scoring. Do not infer, reconstruct, or
    tune against hidden labels.
12. Do not compare surface-space and parcel-space scores as if they were the
    same benchmark. If a bridge projection is used, label it approximate and
    compare only within the same bridge protocol.
