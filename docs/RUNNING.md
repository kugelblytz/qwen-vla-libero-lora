# Running the ten-task experiment

> The completed run used the **fast two-pass** command, resumed from checkpoint 1732,
> and was evaluated with `--budget native`; see [results.md](results.md). The matched
> budget remains the evaluator default, but successes need 976–3,304 simulator steps
> against its 280-step limit. Diagnostic options (`--turn-guard-deg`, history ablation)
> and the training variant are described in [turn-guard.md](turn-guard.md) and
> [state-and-changepoints.md](state-and-changepoints.md).

Use [command.txt](../commands/command.txt) from the project root. Full preparation considers
up to 50 source demonstrations per task, excludes unsuccessful episodes, and
exports a disjoint episode-level train/validation split. The earlier two-per-task
pilot is diagnostic-only and cannot launch production training.

The [fast-training command](../commands/fast-training-command.txt) provides a shorter initial
two-pass option using the optimized kernels, approximately 5–7 hours of compute
after preparation. [command.txt](../commands/command.txt) retains the five-pass option,
approximately 12.5–17 hours. See [efficient-training.md](efficient-training.md)
for the tested environment, measurements and reproduction instructions.

## Initial training

After preparation completes, run the `train-fast --dry-run` command, then the background training
command in `command.txt`. The default is **five nominal passes** with an effective
batch of 32, saving checkpoints every 500 updates and retaining the latest five.
Validation runs every nominal pass. Translation remains 1 cm,
rotation remains 5 degrees, and the model predicts one action at a time.

New exports retain microbatch 16, accumulation 2 and two loader workers as a
compatible default. The latest [kernel benchmark](efficient-training.md) selected
`train-fast --microbatch 32` (accumulation 1). The launcher defaults to four OpenMP
threads unless your environment overrides it. Use `--microbatch 16` for more
memory headroom. Effective batch remains 32 in either case. Explicit overrides
and kernel versions are recorded alongside each launch configuration.

The update budget is `5 × ceil(training_examples / 32)`. Equal task sampling means
these are nominal passes, not exactly five visits to every example. Model vision
and projector weights remain frozen; BF16 rank-64 language LoRA is trained.

In another terminal, run:

```bash
bash scripts/pipeline.sh status
```

This reads the actual run's log and estimates sustained speed after its first ten
updates. No extra GPU computation or throughput benchmark is launched. Preparation,
validation, checkpoint saves and subsequent rollouts add to the total time.

## Resume or extend

Choose an explicit checkpoint from the same run. `--passes` is the **total**
budget, including completed updates, rather than additional passes.

```bash
cd "$(git rev-parse --show-toplevel)"  # repository root
DATASET="$(cat .workspace/libero-object-1cm5deg-dataset.txt)"
# Replace N with the actual checkpoint update number.
CHECKPOINT="$DATASET/training-output/checkpoint-N"
bash scripts/pipeline.sh train-fast --resume "$CHECKPOINT" --microbatch 32 --passes 5 --dry-run
nohup bash scripts/pipeline.sh train-fast --resume "$CHECKPOINT" --microbatch 32 --passes 5 > .workspace/libero-object-training-resume.log 2>&1 < /dev/null &
echo $! > .workspace/libero-object-training.pid
```

Resume restores adapter, optimizer, scheduler and random-number state. Streamed
data replay is not guaranteed bit-identical to an uninterrupted run. If five
passes have finished and validation warrants continuing, use the final checkpoint
and `--passes 10`. Changing the total recalculates cosine/warmup scheduling; this
is a documented staged run, not identical to training ten passes from the outset.
Each invocation preserves its launch configuration and checkpoint provenance in
`training-launches/`. Starting over in a populated output directory is refused.

The scratch verification completed three updates, restored checkpoint 2 and
completed the remaining update. The fresh five-pass export and official parser
also passed; see [verification evidence](../evidence/five-pass-resume-check.json).
The fast environment additionally passed its launcher and checkpoint-resume
checks; see [fast-kernel evidence](../evidence/fast-kernel-benchmark.json).

## Checkpoint selection and evaluation

Review saved checkpoints nearest nominal passes **1, 3 and 5** using held-out action
generation and fixed reserved-demonstration rollouts. Loss alone does not select
the adapter. These selection checks are manual; training does not auto-launch an
evaluation. Checkpoint numbers are optimizer updates, so multiply the desired
pass by `updates_per_nominal_epoch` in `dataset-manifest.json`, then choose an available checkpoint near that update.

For action checks, serve the chosen checkpoint in a separate terminal:

```bash
QWEN_PROFILE=4b-libero-v5 LIBERO_ADAPTER="$CHECKPOINT" bash scripts/sim/serve_qwen.sh
```

Then inspect `pipeline.sh check-checkpoint --help` for the explicit run, adapter,
endpoint and output arguments. Freeze the selected adapter before benchmark runs.
To run ten benchmark attempts per task, with videos:

```bash
bash scripts/pipeline.sh evaluate --run "$DATASET" --adapter "$CHECKPOINT" --attempts 10 --budget matched
```

The evaluator starts its own server; stop the manually served action-check endpoint
first or select a free port. `matched` enforces 280 actual simulator steps; the
separate `--budget native` diagnostic allows 400 semantic decisions. The pilot's
successful conversions exceeded 280 steps, so retain this comparison limitation.


## Diagnostic evaluation options

These change the policy's inputs or choices and are **not** part of the frozen benchmark.
Every output that uses them is marked (`protocol` fields); report them separately.

- `--turn-guard-deg 90` (`evaluate.py`, `eval_episode.py`): withholds yaw beyond ±90° net turn.
  Why, how it works and how to report it: [turn-guard.md](turn-guard.md).
- `--history none|no-rt` (`eval_episode.py`, `history_ablation.py`): changes the recent-moves
  line. Used to test whether the yaw loops depend on the history.

## Training variants

[export_variant.py](../scripts/export_variant.py) re-exports a prepared v5 run with a robot-state
prompt line, a shorter history and changepoint oversampling, without reconverting demos.
Commands and comparison plan: [state-and-changepoints.md](state-and-changepoints.md).
`train.py`, `evaluate.py`, `eval_episode.py` and `check_checkpoint.py` read the prompt file and
history window from the run's `dataset-manifest.json`, so variant runs use the same commands.