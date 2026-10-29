# Robot-state line and changepoint oversampling — training variant

> **Status:** Trained and evaluated (`v6-state-h3-cp4`, 6928 updates). Selected adapter-3500: **63/100** vs
> v5 19/100 under the same frozen protocol. Results: [results-v6.md](results-v6.md).

## Why

Most of the 81 failures of checkpoint 3464 are phase-transition failures: the policy
keeps repeating its previous action (all-yaw loops, pitch loops, holding without
RELEASE). The v5 prompt gives the model only images and the last five MV_/RT_ tokens.
It has no explicit signal of how far it has already turned or whether the gripper is
closed. In the [rotation diagnostic](rotation-prediction-check.md), tomato sauce
predicts the correct first yaw and later translation on expert frames, but loops in
closed loop. That points at weak stopping/transition decisions rather than a wrong
initial label. This is a hypothesis.

Changepoints are also rare in the 1 cm data: the GRASP label share fell from 1.47%
(2 cm single-task) to 0.71%.

## Variant (default `v6-state-h3-cp4`)

`scripts/export_variant.py` derives a new training export from the prepared v5 run,
without reconverting demonstrations:

1. **Robot-state line** in the prompt: `Gripper: open|closed. Turned since start:
   yaw …, pitch …, roll ….` Turns are rounded to whole 5-degree steps and measured
   from the logged TCP pose with the same function rollouts use
   (`contract.turned_since_start`). The prompt becomes `prompt-v6-state-1cm-5deg.txt`.
2. **Shorter recent-moves window:** 3 tokens instead of Show-Harness's 5.
3. **Changepoint oversampling, train split only:** every sample whose label differs
   from the previous label, plus every GRASP/RELEASE/DONE, appears 4 times.

Unchanged: the episode-level train/validation split (validation stays held out and
comparable), images, 1 cm / 5 degrees, and the update budget per nominal pass. The
budget is computed from the un-oversampled training count, so `--passes N` costs the
same number of updates as the v5 run. Each component can be switched off
(`--no-state-line`, `--history-window 5`, `--changepoint-repeat 1`) for ablation.

## Related diagnostics

- `scripts/check_changepoints.py` (`pipeline.sh check-changepoints`): teacher-forced
  check on held-out validation frames, grouped as first / switch / GRASP / RELEASE /
  DONE / continue. Reports accuracy and the **copy rate**, the share of changepoints
  where the model repeats the previous expert label. Works for v5 and variant runs.
  It is not rollout success.
- `scripts/history_ablation.py`: reruns fixed starts (default cream cheese, tomato
  sauce, alphabet soup; states 0–4) with recent-moves history `normal`, `none` or
  `no-rt` on the frozen checkpoint. It tests whether the history list drives the
  loops before training a variant. Diagnostic only, not a benchmark.

## Comparison rule

Compare a variant with checkpoint 3464 at the same update budget, first on the
changepoint check and fixed diagnostic starts, then on the 100-attempt native
evaluation. Report it as a new adapter; the 19/100 frozen result stays unchanged.

The planned v6 run deviates from this rule on purpose: it trains for 6928 updates (4 passes)
instead of 3464, so its result combines the variant with twice the training. State the
budget with every v6 number. See [README.md](experiment-log.md#next-run--v6-against-v5).
