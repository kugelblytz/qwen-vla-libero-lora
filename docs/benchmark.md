# Training configuration benchmark

This is the earlier batch/loader benchmark with the PyTorch Gated DeltaNet
fallback. The [optimized-kernel benchmark](efficient-training.md) supersedes its
speed recommendation: the complete fast path measured approximately 6 s/update.

The benchmark uses the frozen ten-task diagnostic export (1,420 training examples)
with the same Qwen3.5-4B base model, BF16 rank-64 LoRA, frozen vision/projector,
two 256-pixel camera images, 1 cm / 5-degree action labels, effective batch 32,
SDPA, Liger and gradient checkpointing as the training pipeline. It compares
microbatches 4, 8 and 16 with two loader workers, then four workers with the best
fitting batch. Each screening case discards three warm-up updates and measures
eight updates. The baseline and fastest candidate are repeated with five warm-up
and twenty measured updates each.

All cases start fresh with seed 42. Evaluation and checkpoint saves are disabled
during measurements. The callback synchronizes CUDA and measures the intervals
between completed optimizer updates, including data preparation, gradient
accumulation and optimizer work. Peak allocated and reserved memory are recorded
through PyTorch because this MIG allocation does not expose normal GPU utilization
queries. `OMP_NUM_THREADS=4` is used for every case.

Dataset, configuration and per-update timing records are preserved under the
directory named in `.workspace/libero-object-benchmark.txt`. Scratch adapters
are deleted after each case and never used as experiment checkpoints. No
production training is launched or interrupted by this benchmark.

The result selects the fastest stable **tested configuration on this shard**,
not an exhaustive hardware/kernel search. Streamed sample order can differ with
worker count. The small pilot has consistent prompt/image shapes, but its speed
and memory measurements do not guarantee identical behavior on the full dataset
or another machine. Validation, saving and final evaluations add wall time.

The confirmation must improve time per update by more than 3% before recommending
a change from the batch-8 baseline.

## Results

Completed on the H100 MIG shard (26 SMs, 20,096 MiB memory), with a 24-core CPU
quota. All six cases passed; their combined wall time was 29.3 minutes.

| Screening microbatch | Accumulation | Workers | Seconds/update | Peak reserved GiB |
|---|---|---|---|---|
| 4 | 8 | 2 | 19.55 | 12.51 |
| 8 | 4 | 2 | 17.67 | 14.13 |
| 16 | 2 | 2 | 16.88 | 17.43 |
| 16 | 2 | 4 | 16.88 | 17.66 |

The longer confirmation runs measured **17.70 s/update** for batch 8 and
**16.94 s/update** for batch 16, each with two workers. Batch 16 provided **4.48%
higher throughput** (4.29% less time per update), confirming the screening result.
Four workers showed no meaningful improvement in screening.

The selected configuration is **microbatch 16, accumulation 2, two workers**, with
the same effective batch 32. The longer batch-16 run peaked at **18.19 GiB reserved**
and **17.15 GiB allocated**, leaving approximately **1.44 GiB** of the allocation
outside the reserved pool. This headroom is limited; use `train --microbatch 8`
if production data requires more memory. Accumulation then becomes 4 automatically.
Future dataset exports use the selected batch configuration. Existing frozen
exports are preserved; the launcher override is recorded in its launch manifest.

See [compact evidence](../evidence/training-benchmark.json) for all results and the
full local report hash. The training budget remains five nominal passes; this
benchmark changes throughput, not the number of updates or action granularity.
