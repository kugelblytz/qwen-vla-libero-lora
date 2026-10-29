# Subexperiment — Show-Harness 4B, LIBERO-Object, 1 cm

> **Status:** **Postponed (28 October 2026) — not a current priority.** v5 frozen evaluation 19/100; v6 variant (adapter-3500) **63/100** with 4B ([results-v6.md](results-v6.md)) and **61/100** with 2B ([results-v6-2b.md](results-v6-2b.md)) under the same protocol. Further training (grasp recovery, next model iteration) and the matched-budget comparison are paused.
> **Depends on:** The completed single-task 4B training and replay diagnostics

## Current work — 16 October 2026

> **Postponed on 28 October 2026.** This subexperiment and further training are paused and not a
> current priority. Results, evidence and scripts stay as they are; the open rows below are not being worked on.
> When resumed, start from the next steps in [results-v6.md](results-v6.md#6-recommended-next-steps) and
> [results-v6-2b.md](results-v6-2b.md#5-recommended-next-steps).

What we learned from the 100-attempt evaluation is in [results.md](results.md#what-we-learned).
The main failure is phase transition: 45/81 failures never issue GRASP, and 24
attempts repeat RT_YAW_CW for all 400 commands. Successes needed 976–3,304 simulator
steps, so the planned 280-step matched comparison would mostly measure executor
step cost (see [execution budget](results.md#execution-budget)).

| Workstream | Purpose | State |
| --- | --- | --- |
| Yaw-loop pose analysis (`scripts/check_yaw_loops.py`) | Does the wrist actually turn, and when does the arm break away? | Done: all 24 all-yaw attempts (plus 2 others) break away after 26–27 tokens / ~125° ([evidence](../evidence/yaw-loop-poses.json)) |
| History ablation (`scripts/history_ablation.py`) | Do the recent-moves tokens drive the loops? | Implemented; no results recorded |
| Changepoint check (`scripts/check_changepoints.py`, [sweep](../scripts/changepoint_sweep.sh)) | How often does the model copy the previous action at expert changepoints? | Done: v5 0.495 accuracy / 0.472 copy rate; v6 adapter-3500 0.722 / 0.222; final v6 adapter back to 0.486 ([evidence](../evidence/v6/changepoint-sweep/)) |
| [Turn guard](turn-guard.md) | Do looping tasks succeed if over-rotation is withheld? Diagnostic protocol only | Implemented; no results recorded |
| [State line + changepoint oversampling](state-and-changepoints.md) | Training variant `v6-state-h3-cp4`, v5 settings but twice the updates (6928) | **Done: adapter-3500 63/100** vs v5 19/100 (paired McNemar p ≈ 1e-12); all 24 yaw loops gone; final adapter overfits. See [results-v6.md](results-v6.md) |
| 2B with the v6 technique | Show-Harness default backbone (Qwen3.5-2B) with its LoRA recipe and our v6 data, 6928 updates like 4B v6 | **Done: adapter-3500 61/100** vs 4B v6 63/100 (paired McNemar p ≈ 0.85); 16 of 39 failures hold the object but never place it. See [results-v6-2b.md](results-v6-2b.md) |
| Grasp outcome and recovery | All 37 v6 failures lose or miss the object and continue with an empty gripper; no re-grasp | **Postponed:** holding/empty in the state line, recovery training data, executor grasp check ([results-v6.md](results-v6.md#6-recommended-next-steps)) |
| Confidence pilot (for Milestone 04) | Does the 2B v6 adapter's token confidence predict failures early enough to trigger escalation to the cloud? Evaluation only, no training | **Done 28 October 2026: does not support uncertainty as the main M04 trigger.** Offline AUROC 0.74 (changepoint frames) / 0.89 (random frames); live episode AUROC 0.45, 35% of failure events caught vs 86% for random escalation; 61/100 reproduced. See Results |
| Matched 280-step budget | Planned OpenVLA-comparable run | **Postponed.** Open decision: run, redefine, or report the step gap only |
| Checkpoint cadence | Only 1732 and 3464 kept despite `save_steps=500` | Explained by the launch records: the first v5 launch used `save_steps=1732`; `save_steps=500` applied only after the resume. v6 sets cadence explicitly (`--save-steps 250 --keep-checkpoints 3 --snapshot-every 500`) |

The frozen 19/100 result is not changed by any of these. Diagnostic and variant
results are reported separately.

## Completion checklist

- [x] All ten Object tasks have verified converted demonstrations, with exclusions
  and a disjoint training/validation episode split recorded.
- [x] Conversion, training prompts and evaluation consistently use 1 cm translations,
  5-degree rotations and 25 gripper control steps.
- [x] One shared 4B adapter is trained and a checkpoint selected using validation
  action predictions and rollouts, then frozen before benchmark testing.
- [x] Ten native-budget attempts per task are completed and reported, with logs and
  two-camera videos for every attempt, including failures and interruptions.
- [x] Comparison limitations relative to OpenVLA document data, camera, execution-budget and
  training differences alongside per-task success and runtime.

## Goal

Train one Qwen3.5-4B Show-Harness adapter on all ten LIBERO-Object tasks and
measure performance on those same tasks. Use original OpenVLA's suite-level
fine-tuning/evaluation protocol as the comparison reference.

The initial subexperiment is complete when **the frozen adapter has a documented
100-attempt evaluation with videos and explicit comparison limitations**.

## Scope

- Regenerate successful demonstrations into v5 semantic actions at **1 cm**.
- Use a shared rank-64 BF16 language-LoRA adapter with frozen vision/projector.
- Checkpoint selection through reserved validation data and validation rollouts.
- Evaluate the ten Object tasks using standard initial-state indices 0–9.
- Measure actual simulator steps; compare with OpenVLA's 280-step Object horizon.

### Not in scope

- Unseen-task transfer, other LIBERO suites or a direct local OpenVLA run.
- Changing the frozen vision encoder or adding new action/control architectures.
- Claiming 1 cm improves success: existing expert replay showed 9/10 at both sizes.

## Plan

1. **Verify execution and data coverage** — convert/replay one demonstration per
   task; measure simulator-step requirements before preparing the full dataset.
2. **Prepare data** — use OpenVLA's cleaned source selection where provenance can
   be verified; record additional conversion exclusions and split by episode.
3. **Train** — freeze a 1 cm / 5-degree v5 prompt and configuration; start with
   two nominal passes, measuring throughput from the actual run. Compare retained
   checkpoints before deciding whether to extend to five passes. Future checkpoint
   saving is configured every 500 updates; verify cadence after resume.
4. **Select and evaluate** — freeze the validation-selected checkpoint, run 100
   benchmark attempts and save all videos. Report the matched 280-control-step
   horizon separately from any native 400-token diagnostic.
5. **Report** — summarize success, failure reasons, data coverage, latency and cost.
   Expansion to 50 attempts per task (500 total) and three training seeds is an
   optional second stage, after reviewing the initial results.

## Outputs

Task/data manifest, conversion report, frozen prompt/configuration, adapter hash,
checkpoint-selection record, result table and videos. Large artifacts remain local;
compact summaries and provenance belong in Git.

## Comparison limits

This evaluates tasks included in training. OpenVLA's reference uses a 7B model,
external-camera input and smooth control; ours uses a 4B model, two cameras and
converted discrete trajectories. Our validation split and conversion exclusions
also change training coverage. Scores therefore compare complete setups and do
not isolate architecture. A semantic token is not a simulator control step.

See [PLAN.md](PLAN.md) for settings, staging and the official reference sources.
This is a separate follow-up; milestone01's baseline completion criteria remain
unchanged. Training and the 100-attempt native evaluation are complete. See
[results.md](results.md) for per-task outcomes and limitations, [RUNNING.md](RUNNING.md) and
[command.txt](../commands/command.txt) for preparation, training, timing and resume commands.

## Pipeline checks

The 1 cm / 5-degree execution checks passed. The ten-task pilot obtained two
successful conversions per task from 23 candidates (20 successes, two conversion
failures and one original-demonstration failure). Export and official configuration
parsing passed. Successful converted episodes took 992–1,475 active simulator
steps; none fit the 280-step reference horizon. The 19 evaluation successes later
needed 976–3,304. These are demonstration checks,
not learned-policy results. Full preparation subsequently accepted 433/500 episodes;
two-pass training and the final native evaluation are recorded in [results.md](results.md).

The [training benchmark](benchmark.md) selected microbatch 16, accumulation 2 and
two workers: 16.94 seconds/update, 4.48% higher throughput than the batch-8 baseline.
All six cases passed; peak reserved memory in the selected confirmation run was
18.19 GiB. Future exports use this configuration, retaining effective batch 32.

Subsequent profiling found missing optimized Gated DeltaNet/convolution kernels.
The [fast environment](efficient-training.md) measured 6.00 seconds/update with
microbatch 32 and accumulation 1, about 2.8× the prior throughput. Launcher and
resume checks passed. [fast-training-command.txt](../commands/fast-training-command.txt)
provides an initial two-pass option (roughly 5–7 compute hours), preserving all
successful demonstrations, one action per prediction and the 1 cm / 5-degree
interface. The five-pass option remains in `command.txt`.


## v6 run — settings and result

**Result:** adapter-3500, selected by the held-out changepoint check, scored **63/100** against v5's 19/100 with the
same frozen protocol. Yaw loops 24 → 0, GRASP issued in 55 → 100 attempts, RELEASE in 3 → 60. The end-of-run adapter
(6928) had overfitted (validation loss 0.084 → 0.139; changepoint accuracy back to v5 level). Every remaining failure
involves a missed or lost grasp. Full analysis: [results-v6.md](results-v6.md).

The [yaw-loop diagnosis](turn-guard.md#why) found that every all-`RT_YAW_CW` failure drove
the arm away after ~125° of yaw, and that the saturated 5-token history keeps the model turning
([evidence/yaw-loop-poses.json](../evidence/yaw-loop-poses.json)). The v6 variant
([state-and-changepoints.md](state-and-changepoints.md)) uses the settings of checkpoint 3464
with **twice the training length**. Commands: [v6-training-command.txt](../commands/v6-training-command.txt).

| | v5, checkpoint 3464 | v6-state-h3-cp4 |
| --- | --- | --- |
| Episodes, validation split, base model, LoRA (r64, α128, dropout 0.05), frozen vision/projector | as prepared | **identical** (re-exported from the same run) |
| Batch / lr | 32 × 1 / 1e-4 cosine, warmup 0.1 | **identical** (`--microbatch 32`) |
| Seed, task mixing, streaming buffer, cutoff, kernels | 42, equal 10%, 4096, 2048, train-fast | **identical** |
| Updates | 3464 (2 passes × 1732) | **6928** (`--passes 4`); cosine spans all 6928 |
| Prompt | v5 | v5 + `Robot state: …` line and one guidance bullet |
| Recent moves | 5 | 3 |
| Changepoint training rows | 1× | 4× (validation unchanged) |
| Checkpoints | 1732, then every 500 after a manual resume | every 250, newest 3 kept; adapter snapshots every 500 and at 6928 |
| Validation loss (~15 min each) | every pass (1732, 3464) | every pass (1732, 3464, 5196, 6928), as v5; prompts differ, so only roughly comparable to v5 |
| Crash handling | manual resume at 1732 | `train-resilient`: automatic `--resume latest` from the newest complete checkpoint |

The final comparison is v6 at 6928 against v5 at 3464, so it mixes the training change with twice
the updates. v6's adapter at ~3500 is not a like-for-like v5 checkpoint either: its learning rate
has not yet annealed. Report the budget difference with every v6 result.

### Crash recovery

The server is unstable, so training runs under [scripts/train_resilient.sh](../scripts/train_resilient.sh)
(`pipeline.sh train-resilient`):

- Every launch uses `train.py --resume latest`: the newest checkpoint whose files are all present
  and readable (truncated optimizer, scheduler, RNG or adapter files are detected and skipped).
  Adapter, optimizer, scheduler and RNG state are restored, so the schedule continues unchanged.
  Streamed data order after a resume is not bit-identical.
- If the trainer process crashes, the supervisor relaunches after 2 minutes, up to 20 consecutive
  failures without progress. A crash before the first checkpoint moves the partial output aside
  (`training-output-crashed-<time>`) and starts fresh.
- If the whole server restarts, the supervisor dies too: start the same command again. Keep the
  arguments identical, or the schedule changes.
- A second launch on the same run is refused (lock); at most ~250 updates (~25 min) are lost per
  crash. Evaluations resume with `--resume-evaluation`.

Comparison, in order (all in the command file):

1. **Held-out changepoints** (`check_changepoints.py`, no simulator) on both checkpoints, same
   validation episodes: accuracy at expert action changes and the copy rate. The v5 baseline can
   run before training starts.
2. **Frozen-protocol evaluation** of v6 exactly as v5 (10 attempts per task, native budget), then
   `compare_evaluations.py`: per-task success, all-yaw loops, GRASP, RELEASE and step-limit endings.
3. **Optional turn guard** on both checkpoints, reported separately ([turn-guard.md](turn-guard.md)).

Because three things change at once, a v6 improvement shows that the package works, not which
part did it.

### What to watch

The 4× changepoint rows also shift the training label mix (cream cheese, tomato sauce and
alphabet soup): moves 93.8% → 87.8%, turns 4.0% → 6.0% (×1.5), GRASP, RELEASE and DONE
0.7% → 2.1% each (×2.9). v6 may therefore over-predict turns or end early with DONE. Check:

- `check_changepoints`: copy rate at changes should fall; `continue` accuracy should not, and turns
  should not become more frequent among its wrong predictions than for v5.
- `compare_evaluations`: fewer all-yaw loops, more GRASP and RELEASE, and no increase in
  `end:done` without success (v5: 2).
- If turns or DONE are over-predicted, first try post-hoc logit adjustment at inference
  (lower their logits by log 1.5 and log 2.9 via the server's `logit_bias`) before retraining.

Outcome: turns were not over-predicted (3% of actions). The 24 DONE-without-success endings all followed a
missed grasp and a release of the empty gripper, so they are grasp failures, not premature DONE; logit adjustment
would not help.

Frame thinning (dropping interior frames of long same-action runs) was considered and not
included: it would add to the 4× reweighting rather than replace it, and literature gains come
from fewer decisions (waypoints), not fewer training frames. It stays a cheaper alternative
variant (thinning without repetition) if training time becomes the bottleneck.