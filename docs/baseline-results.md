# Baseline and fine-tuning results overview

Neither batch completed its intended 300 attempts. Rates below use completed
attempts only; missing/interrupted attempts are not scored as task failures.
Both runs used prompt v3, temperature 0, a 200-action limit, two camera views,
and the boundary-rendering pipeline. These baseline runs used released adapters.
The local fine-tuning follow-up is reported separately below.

| Run | Training corpus | Completed | Successes | Median rollout duration | Videos |
| --- | --- | ---: | ---: | ---: | ---: |
| 2B sim | ManiSkill + RoboLab | 298/300 | 0/298 | 109.3 s | 298 |
| 4B real | Real Franka + AgileX | 81/300 | 0/81 | 142.9 s | 82 |


## Completed attempts by task

Each entry is successes/completed attempts; the intended count was ten per
task in each batch. A dash means no completed attempt, not zero success.

| Suite | Task | 2B sim | 4B real |
| --- | --- | ---: | ---: |
| libero_spatial | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 0/10 | 0/10 |
| libero_spatial | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 0/10 | 0/1 |
| libero_spatial | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 0/10 | — |
| libero_object | pick_up_the_alphabet_soup_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_cream_cheese_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_salad_dressing_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_bbq_sauce_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_ketchup_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_tomato_sauce_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_butter_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_milk_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 0/10 | — |
| libero_object | pick_up_the_orange_juice_and_place_it_in_the_basket | 0/10 | — |
| libero_goal | open_the_middle_drawer_of_the_cabinet | 0/10 | — |
| libero_goal | put_the_bowl_on_the_stove | 0/10 | — |
| libero_goal | put_the_wine_bottle_on_top_of_the_cabinet | 0/10 | — |
| libero_goal | open_the_top_drawer_and_put_the_bowl_inside | 0/10 | — |
| libero_goal | put_the_bowl_on_top_of_the_cabinet | 0/10 | — |
| libero_goal | push_the_plate_to_the_front_of_the_stove | 0/10 | — |
| libero_goal | put_the_cream_cheese_in_the_bowl | 0/10 | — |
| libero_goal | turn_on_the_stove | 0/10 | — |
| libero_goal | put_the_bowl_on_the_plate | 0/10 | — |
| libero_goal | put_the_wine_bottle_on_the_rack | 0/8 | — |

## Behavior and interruptions

- **2B sim:** {'max_tokens': 255, 'done': 43}; GRASP 3, RELEASE 104, DONE 43. Median recorded HTTP request latency across completed attempts: 0.194 s. Recorded HTTP errors: 0; invalid outputs: 0.
- **4B real:** {'max_tokens': 81}; GRASP 0, RELEASE 0, DONE 0. Median recorded HTTP request latency across completed attempts: 0.248 s. Recorded HTTP errors: 0; invalid outputs: 0.

The 2B run completed Spatial 100/100, Object 100/100 and Goal 98/100. It
stopped during initial-state index 8 of the final Goal task (wine bottle onto
rack); state index 9 was not started. No final summary or exit status was
saved, and the cause is unconfirmed.

The 4B run completed only 81 Spatial attempts: eight tasks × ten attempts,
then state index 0 of the ninth task (bowl next to plate). State index 1 was
interrupted during request-file writing. The traceback, error.json and exit
status are truncated/empty; the underlying exception is unknown. Its 82 videos
include the interrupted attempt. Object and Goal were not reached.

A valid model response and successful control checks establish an operational
pipeline, not successful manipulation or proof that all observation alignment
issues have been eliminated. The 4B run changes both model size and training
corpus, so it is not a controlled scaling comparison.

## Evidence

Local batch directories (large outputs are ignored by Git and are not included
in a fresh clone):

- 2B batch.
- 4B batch.
- [Original protocol](evaluation-protocol.md), pipeline checks, and [setup](setup.md).

## Single-task fine-tuning follow-up — 9 October 2026

These diagnostic results are separate from the released-adapter baseline.

| Follow-up test | Result |
| --- | ---: |
| Fine-tuned Qwen3.5-4B, epoch 1 | 0/10 |
| Same training run, epoch 40 | **6/10** |
| Converted expert replay, 2 cm | 9/10 |
| Converted expert replay, 1 cm | 9/10 |

Epoch-40 evaluation used the same ten benchmark initial states, v5 prompt,
2 cm translations and 200-action limit as epoch 1. All ten videos were saved.
The four failures terminated with DONE without satisfying LIBERO's success
predicate; none issued RELEASE. Epoch 1 was selected by minimum validation
loss; epoch 40 was the retained final checkpoint.

Halving translation steps did not improve expert-replay success and required
approximately 1.9 times as many actions. These replay tests execute expert
commands; no 1 cm policy was trained or evaluated.

See the [follow-up report](single-task-results.md) for the
per-attempt results, preserved summaries and local video locations.

## Suite-level fine-tuning follow-up — 14 October 2026

One shared Qwen3.5-4B BF16 rank-64 language-LoRA adapter was trained on all ten
LIBERO-Object tasks at 1 cm / 5 degrees (433 converted demonstrations, 389 train /
44 validation episodes, 3,464 updates). Checkpoint 3464 was frozen and evaluated on
initial-state indices 0–9 per task with a native 400-command limit. These are
training tasks, not unseen-task transfer.

| Result | Value |
| --- | ---: |
| LIBERO predicate successes | **19/100** |
| Per task | orange juice 8, ketchup 5, alphabet soup 3, salad dressing 3, others 0 |
| End reasons | 19 success, 79 command limit, 2 DONE |
| Failures that never issue GRASP | 45/81 |
| Attempts issuing only RT_YAW_CW | 24 (all cream cheese and tomato sauce) |
| Successes with target still between both finger pads | 17/19 |
| Simulator steps per success | 976–3,304 (OpenVLA Object horizon: 280) |
| Rollout wall time | 6.0 h total; median 63 s success, 239 s failure |

Main lessons: failures are mostly phase transitions (orientation loops, no switch to
translation, no RELEASE), not placement alone; benchmark success differs from a clean
place-and-release; and the 1 cm semantic executor needs several times OpenVLA's step
horizon even when replaying expert commands. Cream cheese fell from 6/10 in the
single-task 2 cm run to 0/10 here, with several settings changed at once.

Diagnostics and a training variant are in progress; see the
[experiment results](results.md) and
[current work](experiment-log.md#current-work).

## Suite-level follow-up, v6 variant — 16 October 2026

The diagnosis above (endless rotation from a saturated 5-move history, rare action changes in
training) led to a variant with a measured robot-state line (gripper open/closed, net turn since
start), a 3-move history and 4× action-change training rows, on the same demonstrations, model and
LoRA settings. It was trained for 6,928 updates; adapter-3500 was selected with a held-out
changepoint check and evaluated with the identical frozen protocol.

| Result | v5 (checkpoint 3464) | v6 (adapter-3500) |
| --- | ---: | ---: |
| LIBERO predicate successes | 19/100 | **63/100** |
| Attempts issuing only RT_YAW_CW | 24 | 0 |
| Attempts issuing GRASP / RELEASE | 55 / 3 | 100 / 60 |
| End reasons | 19 success, 79 limit, 2 DONE | 63 success, 13 limit, 24 DONE |
| Rollout wall time | 6.0 h | 2.4 h |

Paired by task and start, 45 attempts succeed only with v6 and 1 only with v5 (exact McNemar
p ≈ 1e-12). All 37 v6 failures miss or lose the object and continue with an empty gripper; the
model never re-grasps. The end-of-run adapter overfitted. Same caveats as above: training tasks,
not unseen-task transfer. Details: [results-v6.md](results-v6.md).

### v6 on Qwen3.5-2B — 17 October 2026

The same v6 technique on Qwen3.5-2B (Show-Harness 2B recipe, same data and 6,928 updates;
adapter-3500 selected by the same held-out check) scored **61/100** under the identical frozen
protocol, against 63/100 for 4B. Paired by task and start, 48 attempts succeed with both, 13 only
with 2B and 15 only with 4B (exact McNemar p ≈ 0.85). 23 of the 39 failures miss or lose the
object as in 4B; 16 hold it but never place it in the basket, mostly butter and chocolate pudding,
which 4B failed to grasp at all. Details: [results-v6-2b.md](results-v6-2b.md).

## Provisional conclusion

The released adapters did not demonstrate successful transfer in the completed
LIBERO attempts. The locally fine-tuned epoch-40 policy achieved 6/10 successes
on one task, showing that the tested training and execution pipeline can produce
successful manipulation. The epoch-1 comparison shows that aggregate validation
loss alone selected an unsuitable controller. The shared ten-task adapter reached
19/100 on its training tasks, with five tasks at 0/10. This does not establish
reliable performance across LIBERO or reproduce the paper's reported results.

The authors also clarify that the released simulation adapter was intended for
controlled sim-to-real experiments and is not expected to reliably solve
RoboLab benchmark tasks out of the box ([maintainer response](https://github.com/showlab/Show-Harness/issues/2#issuecomment-5716810547)).

The Thor 9B subtest is prepared but has no
reported results yet: three LIBERO-Object tasks × three initial states on the
128 GB Thor, using the real-trained 9B adapter and v3. It checks whether a
larger policy grasps and whether latency is practical; it does not resolve the
training-domain mismatch. Fine-tuning remains follow-up work outside this
short validation milestone.

Full milestone completion remains pending: the baseline is missing two
completed attempts and final batch evidence. Do not label it a completed
300-attempt benchmark or launch more evaluation automatically.
