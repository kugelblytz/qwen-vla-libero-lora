# Results — v6 variant (adapter-3500) vs v5 (checkpoint 3464)

16 October 2026. Both evaluated with the same frozen protocol: 10 standard starts per task,
native budget of 400 decisions, no turn guard. Data: both `summary.json` files, per-step pose
and gripper logs (`rollout/actions.jsonl`), v6 decision logs, final camera frames, and the
held-out changepoint check. Variant: [state-and-changepoints.md](state-and-changepoints.md); training run:
[v6-training-command.txt](../commands/v6-training-command.txt).

Evidence: [v6 evaluation summary](../evidence/v6/v6-3500-evaluation-summary.json),
[frozen evaluation settings](../evidence/v6/v6-3500-frozen-evaluation.json),
[changepoint sweep](../evidence/v6/changepoint-sweep/), [training summary](../evidence/v6/v6-training-summary.json),
[computed metrics](../evidence/v6/v5-v6-metrics.json). Scripts: [analyze_v5_v6.py](../scripts/analyze_v5_v6.py),
[changepoint_sweep.sh](../scripts/changepoint_sweep.sh), [compare_evaluations.py](../scripts/compare_evaluations.py).
Videos, decision logs and pose logs stay local (`<run>/eval-native-20261016T085156/`).

## 1. Headline

| | v5 | v6 |
|---|---|---|
| **Successes** | **19/100** | **63/100** |
| Paired by task and start | 1 solved only by v5 | 45 solved only by v6 (18 by both, 36 by neither) |
| Significance (exact McNemar) | | p ≈ 1e-12 |
| Total evaluation time | 6.0 h | 2.4 h |

| Task | v5 | v6 |
|---|---|---|
| Alphabet soup | 3 | 7 |
| Cream cheese | 0 | **10** |
| Salad dressing | 3 | 6 |
| BBQ sauce | 0 | **9** |
| Ketchup | 5 | **10** |
| Tomato sauce | 0 | 0 |
| Butter | 0 | 2 |
| Milk | 0 | **7** |
| Chocolate pudding | 0 | 2 |
| Orange juice | 8 | **10** |

## 2. What changed in behaviour

| | v5 | v6 |
|---|---|---|
| Pure RT_YAW_CW loops | 24 | **0** |
| Turn tokens, share of all actions | 35.9% | **3.0%** |
| Attempts with > 125° net yaw (break-away zone) | 28 | 7 (all after a lost object) |
| Attempts that issue GRASP | 55 | **100** |
| Attempts that issue RELEASE | 3 | **60** |
| Hit the 400-decision limit | 79 | **13** |
| Median first GRASP (decision) | 52 | 49 |
| Median GRASP → RELEASE | 120 | 86 |
| Median decisions per success | 128 | 126 (114 vs 140 on the 18 starts both solve) |
| Median control steps per success | 1,074 | 1,003 |
| Median seconds per failed attempt | 239 | 72 |

The state line and changepoint weighting did what they were designed for: v6 stops turning,
moves on to approach, grasps and transports. Successful v6 episodes are as long or shorter than v5's,
and failures end quickly instead of burning the full budget.

Held-out changepoint check (teacher-forced, 336 frames per model), for reference:

| | v5 3464 | v6 3000 | **v6 3500** | v6 4000 | v6 6928 |
|---|---|---|---|---|---|
| Accuracy at action changes | 0.495 | 0.646 | **0.722** | 0.651 | 0.486 |
| Copy rate at changes (lower = better) | 0.472 | 0.259 | **0.222** | 0.297 | 0.472 |
| Continue-step accuracy | 0.90 | 0.76 | 0.79 | 0.81 | 0.91 |

The offline ranking predicted the rollout result: 3500 is best, and the end-of-training adapter (6928)
had lost the gain (overfitting; validation loss rose from 0.084 to 0.139 after 3464).

## 3. Why v6 still fails (37 attempts)

**All 37 failures happen with an empty gripper.** Gripper opening right after GRASP is 0.039–0.046 m
in every success (fingers stopped by the object) and 0.001 m (fully closed on nothing) in the failures:

| Failure | Count | What the logs show |
|---|---|---|
| Missed grasp, then "complete" task | 24 | Fingers close on nothing (0.001 m). The model carries the empty gripper to the basket, releases ~8 decisions later and says DONE. The wrist camera shows an empty basket; the object is still on the table. Butter 8, chocolate pudding 8, alphabet soup 3, salad dressing 3, bbq sauce 1, milk 1. |
| Missed or slipped grasp, then wander | 13 | Tomato sauce: 5 attempts close on nothing, 5 get a partial grip (0.013–0.033 m) and lose the bottle within ~30 decisions; 2 milk and 1 salad dressing attempts hold the object at first and lose it later (opening ≤ 0.004 m at the end). With nothing in hand the model never reaches a release condition: ~160–240 MV_RIGHT moves (mostly 3–7 mm, i.e. against limits) and 110–162° of drift in yaw until the budget ends. |

Consequences:
- **The earlier "early DONE" reading was wrong.** These are not premature DONEs: the model completed its
  sequence correctly for what it believed was a held object. Down-weighting DONE (logit adjustment) would
  not help; the object is already lost.
- **The model never re-grasps** (0 attempts with more than one GRASP). Expert demonstrations never miss, so
  recovery behaviour is absent from the training data.
- **The state line says `Gripper: closed` whether or not something is held**, so the prompt cannot tell the
  model the grasp failed. The wrist image shows it, but the model does not use that cue.
- Flat boxes (butter, chocolate pudding) and the tomato-sauce bottle are the hard grasps; most other objects
  succeed in ≥ 7/10.

## 4. Other observations

- **27 of 63 successes still hold the object at the end** (36 release it). LIBERO's predicate counts
  an object held over the basket; a real handover task would not.
- **The stuck guard fired in 245 decisions**, mostly in the empty-gripper wandering phase.
- **No direction ping-pong in either model** (median 0 immediate reversals per 100 decisions).

## 5. Caveats

- adapter-3500 is a mid-schedule snapshot (learning rate ~58% of peak) chosen with the held-out check from
  a 6928-update run whose final adapter was worse. It is not a settled model.
- Evaluation tasks are the training tasks (standard LIBERO practice); this shows the fixes work, not
  generalisation.
- One rollout per start; per-task differences of 1–2 successes are noise. The overall paired difference is not.

## 6. Recommended next steps

> **Postponed (28 October 2026):** these next steps are paused and not a current priority.

1. **Make grasp outcome visible.** Extend the state line with what the gripper holds, measured from finger
   opening (e.g. `Gripper: closed, holding object` vs `Gripper: closed, empty`). Same function in export and
   evaluation, as for the turn state.
2. **Teach recovery.** Add training examples where a grasp misses: from converted demonstrations, synthesise
   "closed on nothing → RELEASE → re-approach → GRASP" segments, or record failed grasps from rollouts and
   label the recovery. Without such data the model cannot learn to retry.
3. **Executor-level grasp check (no retraining):** after GRASP, if the opening is below ~5 mm, report a failed
   grasp (and optionally reopen). Useful as a diagnostic now and as a safety net later.
4. **Retrain at the v5 budget (2 passes, 3,464 updates)** so the selected point is a settled model, with
   recovery data and the holding state included. Keep the 4× changepoint weighting; the DONE weighting is not
   the problem.
5. **Grasp alignment for flat boxes and the tomato bottle:** inspect videos of near-misses
   (attempt folders `task-06-*`, `task-08-*`, `task-05-*` of the evaluation) for systematic offset (e.g. stopping one 1 cm step short).
