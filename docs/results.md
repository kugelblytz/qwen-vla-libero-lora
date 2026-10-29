# Results — LIBERO-Object 4B, 1 cm / 5 degrees

> These are the v5 (checkpoint 3464) results. The follow-up v6 variant scored 63/100 with the same
> protocol; see [results-v6.md](results-v6.md). The same variant on Qwen3.5-2B scored 61/100; see
> [results-v6-2b.md](results-v6-2b.md).

## Completed native evaluation

Checkpoint 3464 achieved **19/100 LIBERO task-predicate successes (19%)** across all ten Object tasks, with ten standard initial-state indices per task. All 100 completed attempts have two-camera videos. Pauses, failed startup and a result-collection bug were handled by preserving completed attempts and rerunning interrupted attempts; only complete attempts are counted.

| Target | Successes |
| --- | --- |
| Alphabet soup | 3/10 |
| Cream cheese | 0/10 |
| Salad dressing | 3/10 |
| Bbq sauce | 0/10 |
| Ketchup | 5/10 |
| Tomato sauce | 0/10 |
| Butter | 0/10 |
| Milk | 0/10 |
| Chocolate pudding | 0/10 |
| Orange juice | 8/10 |

This is the **native 400-semantic-command evaluation**, not the planned matched 280-actual-simulator-step comparison. A semantic command can execute multiple simulator steps. These results do not reproduce OpenVLA's score or isolate architecture differences.

End reasons: 19 success, 79 reached the 400-command limit, 2 issued DONE without success (chocolate pudding, initial states 0 and 6). The 100 attempts took 6.0 hours of rollout wall time; median 63 s for successes and 239 s for failures.

### Execution budget

All 19 successes needed **976–3,304 actual simulator control steps** (median about 1,050), using 120–256 semantic commands at about 8–13 control steps per command. The count is every `env.step` after the initial-state reset (`ControlCounter` in `scripts/contract.py`); no success ended inside a partial command. This is inherent to the 1 cm executor, not to the policy: original Object demonstrations take about 136–179 raw steps, and their converted 1 cm replays needed about 1,165–1,839 ([conversion outcomes](../evidence/preparation-summary.json)).

A 280-step matched budget therefore allows roughly 30–35 commands, while the shortest success needed 120. **Under the planned matched protocol this checkpoint, and the converted expert demonstrations themselves, would be expected to score near 0/100.** Such a result would measure the executor's step cost, not decision quality. Whether to run it, redefine it, or report only the step gap is an open decision (see [PLAN.md](PLAN.md)).

## Training and data

All 500 original source candidates were checked. Thirty-three failed original-action replay, and another 34 failed converted replay; 433 converted episodes passed (419 simplified paths, 14 full-path fallbacks). The episode-level split retained 389 training episodes / 55,411 examples and 44 validation episodes / 6,305 examples. Training uses the frozen v5 prompt, 1 cm translations, 5-degree rotations, 25-step gripper commands, two cameras and recent action history.

One shared Qwen3.5-4B BF16 rank-64 language-LoRA adapter was trained for two nominal passes (3,464 updates, effective batch 32); vision and projector remained frozen. Validation loss improved from 0.06768 at checkpoint 1732 to 0.05632 at checkpoint 3464. The final checkpoint was frozen for evaluation. Nominal passes reflect the configured update budget and equal-task streaming, not guaranteed exact exposures to every example: the trainer itself logged epoch 0.5 at checkpoint 1732 and **epoch 1.36 at checkpoint 3464** ([training summary](../evidence/training-summary.json)). The plan's five-pass budget was reduced to two for this first run.

Optimized Gated DeltaNet and causal-convolution kernels cut measured training time from about 17 to 6 seconds/update. The interrupted initial run lost unsaved work; resume from checkpoint 1732 completed successfully. Saving every 500 updates was configured subsequently, but only checkpoints 1732 and 3464 were retained in the completed run. The resumed checkpoint cadence still needs verification before further training.

## What we learned

- Forty-five of 81 failed attempts never issue GRASP. Twenty-four episodes issue RT_YAW_CW for every command, including all cream-cheese and tomato-sauce attempts. These are approach/orientation failures, so placement alone is not the problem.
- Thirty-six failures issue GRASP, but a GRASP label does not establish physical grasp success. Reviewed videos show both useful target grasps followed by failed transport/placement and repeated movements or loss of the target.
- Only three episodes issue RELEASE, and only one is a benchmark success with an explicit release. Reconstruction of all 19 successful final simulator states reproduced the predicate results; 17/19 still had contact between the target and both finger pads. Benchmark predicate success must be distinguished from a completed place-and-release sequence. See [final-state checks](../evidence/all-success-final-states.json).
- A three-state held-out pre-grasp test passed 3/3 with matching expert controls. A separate 48-frame expert-state prediction check found task-dependent difficulty switching out of rotations. Tomato sauce's initial yaw choices match the demonstrations, and it can predict translation on expert states; its live loop therefore needs an execution/observation and phase-transition diagnostic.
- Rotation loops are not limited to yaw: milk never issues GRASP, and 7/10 milk attempts are dominated by RT_PITCH_BACK.
- The [yaw-loop pose analysis](../evidence/yaw-loop-poses.json) (`scripts/check_yaw_loops.py` on the logged poses of all 100 attempts) found that every all-RT_YAW_CW attempt drove the arm more than 10 mm away from its position after 26–27 tokens (124–131 degrees of yaw); the controller no longer holds the TCP and the attempt cannot recover. Two further attempts (alphabet soup state 5, butter state 4) broke away the same way during a long initial yaw run before switching to translation: **26 break-away attempts in total**. About 58–74% of yaw commands in these runs produced a measurable turn. According to `scripts/contract.py`, expert demonstrations never exceed 70 degrees of net yaw; that comparison is not part of this evidence file.
- Cream cheese scored 0/10 here versus 6/10 for the single-task [v5 epoch-40 policy](single-task-results.md). Those runs differ in step size (2 cm / 10 degrees there), task mixing and training repetition (40 single-task epochs), so this is a regression in outcome, not an isolated cause.
- Longer training and targeted transition sampling are reasonable follow-ups, but neither is established as the fix. The 1 cm representation reduces the GRASP label share from 1.47% in the earlier single-task 2 cm dataset to 0.71% here; this comparison also changes task coverage and training repetition.

## Comparison limits and next work

The source is original LIBERO Object demonstrations, not an exact verified match to OpenVLA's cleaned selection. Conversion exclusions, episode holdout, model pretraining, LoRA settings, image inputs, action representation, training budget and execution horizon differ. No local OpenVLA baseline or matched-budget evaluation has been run for this adapter. Training-task performance does not establish unseen-task generalization.

Next work is tracked in [README.md](experiment-log.md#current-work). In short:

1. Finish the rotation-loop diagnostics: the yaw-loop pose analysis is done; a history ablation on fixed starts and a held-out changepoint check have not run yet.
2. A diagnostic [turn guard](turn-guard.md) evaluation, reported separately from the frozen benchmark.
3. A [state-line and changepoint-oversampling training variant](state-and-changepoints.md) at the same update budget as v5.
4. Extending training (five nominal passes) remains an option on fixed diagnostic starts before any second full evaluation.
5. A separate release-and-stable-placement diagnostic, without changing the frozen LIBERO metric.

## Evidence and local videos

[Full per-attempt results](../evidence/full-evaluation-summary.json), [frozen evaluation selection](../evidence/full-evaluation-selection.json), [training summary](../evidence/training-summary.json), [dataset manifest](../evidence/dataset-manifest.json) and [conversion outcomes](../evidence/preparation-summary.json) are versioned. [Rollout review](rollout-review.md), [grasp check](grasp-check.md) and [rotation diagnostic](rotation-prediction-check.md) document their limited scopes.

In this JupyterLab workspace, videos are under full-eval-videos/task-XX-attempt-YYY/trajectory.mp4; external camera is left and wrist camera right. Those folders link to local .workspace artifacts and are excluded from Git, as are weights and full simulator states. A fresh clone contains the summaries and scripts, not the videos or model files.
