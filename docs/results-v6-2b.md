# Results — v6 technique on Qwen3.5-2B (adapter-3500) vs 4B v6 (adapter-3500)

17 October 2026. The 2B run uses the v6 technique (state line, 3-move history, action-change rows ×4) on
Qwen3.5-2B, the Show-Harness default backbone, with the Show-Harness 2B training recipe
([v6-2b-training-command.txt](../commands/v6-2b-training-command.txt)). Same data split, same 6,928 updates as the 4B v6 run,
and the same frozen protocol: 10 standard starts per task, native budget of 400 decisions, no turn guard.
The adapter was selected unattended by [sweep_then_evaluate.sh](../scripts/sweep_then_evaluate.sh).

Evidence: [2B evaluation summary](../evidence/v6-2b/v6-2b-3500-evaluation-summary.json),
[frozen evaluation settings](../evidence/v6-2b/v6-2b-3500-frozen-evaluation.json),
[changepoint sweep](../evidence/v6-2b/changepoint-sweep/), [training summary](../evidence/v6-2b/v6-2b-training-summary.json),
[computed metrics](../evidence/v6-2b/v6-4b-v6-2b-metrics.json). Script: [analyze_v6_2b.py](../scripts/analyze_v6_2b.py).
Pose and gripper logs were kept for the 39 failed attempts only; videos stay on the GPU server
(`<run>/eval-native-*`).

## 1. Headline

| | 4B v6 | 2B v6 |
|---|---|---|
| **Successes** | **63/100** | **61/100** |
| Paired by task and start | 15 solved only by 4B | 13 solved only by 2B (48 by both, 24 by neither) |
| Significance (exact McNemar) | | p ≈ 0.85 (no detectable difference) |
| Total evaluation time | 2.4 h | 2.6 h |

| Task | 4B v6 | 2B v6 |
|---|---|---|
| Alphabet soup | 7 | 7 |
| Cream cheese | 10 | 7 |
| Salad dressing | 6 | 8 |
| BBQ sauce | 9 | 9 |
| Ketchup | 10 | 10 |
| Tomato sauce | 0 | 0 |
| Butter | 2 | 1 |
| Milk | 7 | 6 |
| Chocolate pudding | 2 | 3 |
| Orange juice | 10 | 10 |

**The v6 technique carries over to the 2B backbone.** Half the parameters give the same success rate; the
per-task differences are within the 1–2 attempt noise of one rollout per start.

## 2. Behaviour

| | 4B v6 | 2B v6 |
|---|---|---|
| Pure RT_YAW_CW loops | 0 | 0 |
| Turn tokens, share of all actions | 3.0% | 2.9% |
| Attempts with > 125° max yaw | 7 | 9 |
| Attempts that issue GRASP | 100 | 100 |
| Attempts that issue RELEASE | 60 | 66 |
| Attempts with more than one GRASP | 0 | 0 |
| Hit the 400-decision limit | 13 | **21** |
| Median first GRASP (decision) | 49 | 50 |
| Median GRASP → RELEASE | 86 | 75 |
| Median decisions per success | 126 | 122 (120 vs 124 on the 48 starts both solve) |
| Median control steps per success | 1,003 | 1,003 |
| Successes that release the object | 36/63 | 45/61 |
| Median seconds per success / failure | 61 / 72 | 51 / 195 |

Successful episodes look the same for both sizes. 2B ends more successes with a RELEASE, and its failures last
longer because more of them run into the decision limit (section 3).

Held-out changepoint check (teacher-forced, same frames as for 4B), used to select the adapter:

| 2B snapshot | 2500 | 3000 | **3500** | 4000 | 4500 | 6928 |
|---|---|---|---|---|---|---|
| Accuracy at action changes | 0.528 | 0.670 | **0.679** | 0.542 | 0.557 | 0.420 |
| Copy rate at changes (lower = better) | 0.425 | 0.274 | **0.255** | 0.387 | 0.387 | 0.528 |
| Continue-step accuracy | 0.84 | 0.79 | 0.89 | 0.86 | 0.89 | 0.93 |

The same pattern as 4B: the best snapshot is mid-schedule (3500, 4B: 0.722 / 0.222), and the end-of-training
adapter is worse than v5 was. Validation loss is lowest at update 3464 (0.0898) and rises to 0.125 at 5196.
At 3500 the 2B adapter is weaker than 4B on GRASP changepoints (0.82 vs 0.98) and DONE (0.61 vs 0.75), and
stronger on RELEASE (0.66 vs 0.43).

## 3. Why 2B fails (39 attempts)

Classified from the gripper opening right after the first GRASP (≤ 5 mm: closed on nothing; 5–35 mm: partial;
≥ 35 mm: fingers stopped by the object, as in every 4B success):

| Failure | Count | What the logs show |
|---|---|---|
| Missed grasp (empty or partial), then DONE | 15 | As in 4B: fingers close on nothing or barely grip, the model completes its sequence and says DONE. Butter 4, tomato sauce 3, cream cheese 3, salad dressing 2, milk 2, bbq sauce 1. |
| Missed grasp, then wander to the limit | 7 | Empty gripper, no release condition reached. All tomato sauce. |
| Lost after a firm grasp | 1 | Butter: held (39 mm), empty at the end. |
| **Held, never released** | **13** | **New in 2B.** Firm grasp (39–55 mm) kept to the end, no RELEASE. After GRASP the model issues 95–258 MV_UP and 86–210 MV_RIGHT tokens, lifting the object to 0.34–0.86 m, until the budget runs out. Chocolate pudding 7, butter 4, milk 2. |
| **Held, released outside the basket** | **3** | **New in 2B.** Alphabet soup: firm grasp (57–63 mm), one RELEASE away from the basket, then DONE. |

Consequences:
- **2B grips flat boxes that 4B missed, then cannot place them.** Butter and chocolate pudding failed in 4B because the
  grasp missed (8 + 8 attempts). In 2B, 11 of their 16 failures hold the object and overshoot upward and to the right
  instead of descending into the basket. The bottleneck for these objects moves from grasping to placing.
- **Tomato sauce stays at 0/10** for both sizes: 9 of 10 2B attempts close on nothing.
- **Empty-gripper failures remain the largest group** (23 of 39), so the recommendations of
  [results-v6.md](results-v6.md) (holding state in the state line, recovery data, executor grasp check) apply unchanged.
- **The model never re-grasps** (0 attempts with more than one GRASP), as with 4B.

## 4. Caveats

- adapter-3500 is a mid-schedule snapshot chosen with the held-out check, as for 4B; it is not a settled model.
- Evaluation tasks are the training tasks; this compares model sizes, not generalisation.
- One rollout per start; the paired difference between the sizes is not significant, and per-task differences are noise.
- Gripper and pose logs exist for failures only, so successes cannot be compared step by step.
- The local copy of the training log ends at update 6100; the final validation loss is not recorded.

## 5. Recommended next steps

> **Postponed (28 October 2026):** these next steps are paused and not a current priority.

1. **Use 2B as the default size for the next iterations.** Same success, faster training (about 3.4 s vs 4.2 s per
   update) and closer to the Show-Harness setup. Confirm with the next retrain that both sizes stay level.
2. **Apply the v6 next steps to 2B** (holding state, recovery data, executor grasp check, retrain at 3,464 updates).
3. **Inspect the 13 held-never-released videos** (chocolate pudding, butter, milk) to see whether the model fails to
   see the basket from above or misreads the object in hand as not yet lifted. Placement for flat boxes may need
   more transport-phase examples once the grasp works.
