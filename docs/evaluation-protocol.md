# Evaluation protocol (released-adapter baseline)

Measure transfer of the released ManiSkill/RoboLab-trained adapter to standard LIBERO. This is a small descriptive baseline, not a statistical benchmark or reproduction of the paper.

## Fixed protocol

| Setting | Requirement |
|---|---|
| Policy | Released `qwen3_5_2b_sim` adapter; no further training or tuning |
| Tasks | All 30 tasks: 10 each in standard `libero_spatial`, `libero_object`, and `libero_goal`; preserve exact task names and BDDL definitions from the pinned checkout |
| Episodes | 10 attempts per task, provided initial-state indices 0–9: 300 attempts total |
| Reset randomness | Seed 0 for each task; preserve actual fixture poses alongside the provided initial state |
| Instruction | Task sentence read from the task's BDDL by the upstream runner |
| Prompt | `--version v3`, matching the released adapter |
| Rendering | CPU OSMesa; render both views at semantic decision boundaries, deferring unused intermediate camera images; retain all physics/control steps |
| Motion | Upstream LIBERO executor and camera configuration, `--step-m 0.02` |
| Limits | 200 semantic decisions per episode; 120 seconds per VLM request |
| Decoding | Temperature 0; preserve serving chat template and resolved output constraints |
| Stop behavior | Honor `DONE`; do not enable `--ignore-done` |
| Human input | No manual action selection or takeover |

Use run_full_eval.sh, which invokes the pinned upstream episode loop with evidence capture for the exact 30 tasks in [evaluation-tasks.json](../evidence/setup/evaluation-tasks.json). It runs initial-state indices 0–9 for each task. Freeze revisions, task list, and resolved settings before the batch. Pilot/smoke-test runs are separate. Do not replace failed attempts or select favorable initial states; retain earlier batches if a setup correction requires rerunning, unless the user explicitly requests their removal. The cancelled slow-rendering batch was removed at the user’s request before this restart.

## Outcome handling

Success is determined only by LIBERO's task success predicate. Model `DONE` without environment success is failure. Stop on success, `DONE`, the action limit, environment termination, or an unrecovered error. Preserve upstream invalid-output and timeout/retry behavior, recording it before evaluation rather than introducing a new fallback.

The released adapter's action vocabulary contains six `MV_*` translations, `GRASP`, `RELEASE`, and `DONE`. Document how the LIBERO runner handles this vocabulary and any output constraints or recovery actions. Additional runner capabilities do not establish that the released adapter was trained to use them.

Keep startup, reset, rendering, and server errors visible as infrastructure failures. An infrastructure-only attempt does not satisfy evaluation of that task; resolve the blocker and retain the failed attempt when rerunning. Valid closed-loop episodes that fail manipulation do count, including a batch with zero successes.

## Minimal evidence

For each attempt preserve task name, seed, initial-state index, fixture poses, action sequence/count, outcome/end reason, per-request latency, total duration, invalid outputs, and timeouts. Save a side-by-side agent/wrist video for every rollout, including failures, and highlight a successful example if available. Videos include the final observation and play four semantic decisions per second.

Classify each failed attempt as manipulation failure, premature `DONE`, action limit, invalid output, timeout, simulator/environment failure, server failure, or unknown; add a short diagnosis where supported by the trajectory.

Create a short `results.md` containing:

- a 30-row table of task, outcome, action count, and failure reason;
- total successes out of 300 and per-task successes out of 10, aggregate success rate, median request latency, and mean action count;
- invalid-output, timeout, and failure-category counts;
- links to pinned setup, resolved configuration, logs, and a representative run;
- the limitation that ten provided initial states per task give a limited descriptive estimate, not a robust benchmark and that the adapter was not trained on LIBERO.

Completion requires a functioning policy loop and complete evaluation evidence for all 30 tasks and 300 attempts. No minimum success rate or comparison with published numbers is required. Further episodes, fine-tuning, and optimization belong to later work.
