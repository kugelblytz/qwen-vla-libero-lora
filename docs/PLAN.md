# Plan: Show-Harness 4B on LIBERO-Object with 1 cm actions

Status: steps 1–5 completed for the first stage, with deviations below; this file keeps
the original plan. Current work is tracked in [README.md](experiment-log.md#current-work).

**Deviations from this plan (14 October 2026):**
- Training used **two** nominal passes (3,464 updates; the trainer logged epoch 1.36),
  not five. Only checkpoints 1732 and 3464 were retained, so the planned pass 1/3/5
  comparison could not be made.
- The source selection is original LIBERO Object demonstrations, not a verified match
  to OpenVLA's cleaned selection (433 converted episodes passed).
- Only the **native** 400-command evaluation was run: 19/100. All successes needed
  976–3,304 simulator steps. The matched 280-step budget in step 5 is not reachable by
  the 1 cm executor even with expert commands, so it would not measure decision
  quality. Whether to run, redefine or drop it is an open decision.
- The 500-attempt and three-seed extensions were not started.

## Question and scope

Can one Qwen3.5-4B Show-Harness adapter learn all ten LIBERO-Object tasks?
Measure same-task execution after fine-tuning, using the original OpenVLA
LIBERO-Object experiment as a reference. Unseen-task transfer, other suites and running OpenVLA locally are separate
follow-ups. This experiment uses 1 cm translations throughout.

OpenVLA trained each suite independently: 50 supplied demonstrations per task,
454 successful Object demonstrations after filtering, 50 evaluation attempts
per task and three seeds. It used the external camera only. Its Object score
was 88.4%. These settings concern original OpenVLA, not OpenVLA-OFT.

## Steps

1. **Freeze the protocol and check the execution budget.** Pin code, datasets,
   base model and conversion settings; enumerate all ten task names and source
   episode IDs. OpenVLA allows 280 active simulator control steps for Object,
   preceded by ten settling steps. Instrument every simulator step inside
   Show-Harness's executor; a semantic token can consume many control steps.
   Check expert conversion/replay on one episode per task against this budget
   before committing to full preparation. If the semantic executor cannot
   complete demonstrations within it, retain and report this limitation.

2. **Prepare the demonstrations.** Begin with OpenVLA's cleaned Object source
   episode selection, validating provenance against the original HDF5 data.
   Regenerate Show-Harness v5 observations and labels at 1 cm / 5 degrees,
   with 25 gripper steps and both 256-pixel camera views. Derive and freeze an
   experiment-local v5 prompt explicitly describing 1 cm translations and 5-degree rotations. Use the
   identical prompt, `step_m=0.01` and `rot_step_deg=5` in conversion, training
   and evaluation;
   check prompt hashes and metadata before each stage. Regenerate data rather
   than relabeling the old 2 cm images. Failed conversions remain diagnostics,
   never successful training examples. Use a recorded conversion cap of 600
   tokens; report truncation separately from manipulation failures.
   Verify task success,
   pre-action observation alignment and terminal images. Save exclusions per
   task: the 454 reference episodes are an input target, not a promised number
   of successful conversions. Split complete source episodes approximately
   90/10 per task for training/validation, with no shared source IDs. Report
   overlap with benchmark evaluation starting states rather than claiming
   an independently sampled state holdout.

3. **Train one shared adapter.** Start fresh from Qwen3.5-4B; mix all ten tasks
   with equal task sampling. Use the proven BF16 language-LoRA configuration:
   rank 64, alpha 128, dropout 0.05, frozen vision/projector, learning rate
   1e-4, cosine schedule, 10% warmup and effective batch 32. Keep v5 and the
   non-thinking template aligned. Start with **five nominal passes**, measuring
   sustained speed from the actual run after the first ten updates; no separate
   two-hour throughput benchmark. Save checkpoints each nominal pass and evaluate
   passes 1, 3 and 5. Extend only after reviewing validation rollouts. This is our
   Show-Harness recipe, not a reproduced OpenVLA training recipe. Nominal passes
   define an update budget under equal task sampling, not exact per-example epochs.

4. **Select and freeze a checkpoint.** Use held-out free-generation action
   checks, including GRASP/RELEASE and direction changes. Compare candidate
   checkpoints with fixed validation rollouts initialized from reserved
   demonstration starts. Record aggregate loss but do not select solely by it.
   Freeze the adapter hash before benchmark testing. Reserve final benchmark
   outcomes for reporting rather than further checkpoint tuning.

5. **Evaluate in stages, recording every video.** First run ten standard
   initial states per task (100 attempts). If the pipeline completes correctly,
   extend to indices 0–49, reusing the first ten results when checkpoint and
   settings are unchanged (500 attempts total). Match task instructions,
   success predicates and initialization/settling with the pinned OpenVLA
   runner. For budget-matched results, enforce 280 actual active simulator
   steps, including movements, gripper commands and settling inside execution;
   stop inside an atomic command at the limit. If necessary, run a separately
   labeled native-budget diagnostic with a 400-token limit, doubled from the
   prior 2 cm experiment to allow additional decisions. This does not guarantee
   equivalent physical duration; report actual simulator steps in both. Never
   equate these two budgets or compare their scores as identical protocols.

6. **Report and decide whether to repeat.** Report per-task and suite success,
   confidence intervals, termination reasons, conversion coverage, simulator
   steps, semantic decisions, latency, training time and memory. Preserve logs,
   configuration, hashes and two-camera videos for every attempt. One training
   seed is the initial result; three independent training seeds with 500
   attempts each are the optional published-protocol-scale extension.

## Comparison boundaries

The ten-episode pilot showed 9/10 expert-replay success at both 1 cm and 2 cm,
with approximately 1.9 times as many actions at 1 cm. That earlier pilot retained
10-degree rotations. The new 1 cm / 5-degree motion checks passed, and the ten-task
conversion check produced 20 successful episodes from 23 candidates. Successful
conversions needed 992–1,475 simulator steps, exceeding the reference horizon.
Finer steps have not yet shown better policy performance. The broader dataset's training duration and
sequence lengths must be measured; do not extrapolate the single-task runtime
without the preparation report.


The published 88.4% is reference context. Our two-camera input, regenerated
trajectories, 90/10 split, exclusions, 4B VLM initialization and training recipe
all differ. OpenVLA starts from a robot-action-pretrained 7B model. Budget-matched
execution improves comparability but does not isolate architecture. A controlled
head-to-head would require running both systems with an agreed common source
split and evaluation protocol. If only native-budget results are available,
label them explicitly as an adapted Show-Harness benchmark.

## Outputs and next action

Produce a task/data manifest, frozen training configuration, preparation report,
checkpoint-selection record, evaluation summary and videos. Keep this a separate
follow-up under milestone01; baseline completion criteria are unchanged.

Superseded: preparation, two-pass training and the native evaluation are complete.
See [results.md](results.md) and [current work](experiment-log.md#current-work).

## Sources

- [OpenVLA paper, Appendix E](https://arxiv.org/html/2406.09246v2#A5.SS1).
- [Official training/evaluation overview](https://github.com/openvla/openvla#libero-simulation-benchmark-evaluations).
- [Official evaluation runner: Object horizon and initialization](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py).
