# Faster training without changing the data or rollout actions

The earlier trainer lacked FLA and causal-conv1d, so Qwen3.5 used its PyTorch
Gated DeltaNet fallback. Installing optimized kernels in an isolated environment
reduced measured update time from **16.94 to 6.00 seconds**, about **2.8× higher
throughput**, without changing tasks, images, labels, prompts, LoRA targets,
effective batch 32, learning rate or action granularity.

## Measurements

| Kernel configuration | Microbatch / accumulation | Seconds/update | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---|---|
| PyTorch fallback | 16 / 2 | 16.94 | 17.15 | 18.19 |
| FLA; PyTorch convolution | 16 / 2 | 7.31 | 13.73 | 14.96 |
| FLA; PyTorch convolution | 32 / 1 | 6.94 | 16.46 | 18.89 |
| FLA + causal-conv1d | 16 / 2 | 6.26 | 13.62 | 14.74 |
| FLA + causal-conv1d | 32 / 1 | **6.00** | **16.24** | **18.66** |

All used two loader workers and four OpenMP threads on the same H100 MIG
allocation (26 SMs, 20,096 MiB). The complete-fast-path cases discarded three
warm-up updates and measured eight updates each, starting fresh with seed 42.
They used the same frozen ten-task pilot export. Initialization and first-use
kernel compilation are excluded from steady-state timings. The earlier fallback
confirmation measured twenty updates after five warm-up updates.

Batch 32 was fastest and fit the test. Reserved memory includes reusable cached
memory, so it is not the same as live tensor allocation. Batch 16 provides more
headroom with a small speed penalty. Full-data shape/ordering variation may change
timings or memory; the actual run's status command supplies updated estimates.

## Environment and checks

`.workspace/lf-fast-venv` inherits the pinned trainer and Torch 2.11.0+cu129
packages from the original environment, and contains FLA 0.5.2, Triton 3.7.1 and
causal-conv1d 1.7.0. This isolates the compiler override from the original
Torch/Triton 3.6 environment, which remains usable as the slower fallback.
The newer compiler is required because FLA rejects Triton 3.4–3.7.0 on Hopper
for a known gated backward-kernel correctness issue. This is an explicitly
tested compiler override, rather than Torch's default package dependency set.

Convolution was built against the existing Torch using a local, hash-verified
CUDA 12.9 toolkit. Only its build architecture list was changed to the detected
GPU; kernel source is unchanged. No system CUDA installation was modified.

Recreate the optional environment from the project root:

```bash
bash scripts/pipeline.sh setup-fast --causal
```

On another platform or CUDA version, provide `--cuda-home /path/to/matching/toolkit`
when building the convolution. Plain `setup-fast` installs FLA without building
the convolution and enables the partial speedup. The original trainer setup must
exist first; the fast environment is an overlay and depends on it.

Gated DeltaNet BF16 output relative-L2 differences were below 0.6%, and gradient
differences below 0.9%, against the Torch implementation on the tested shapes.
Convolution checks passed for FP32 and BF16; BF16 output differences were below
0.3% and gradient differences below 0.5%. These check numerical agreement on
representative small shapes, not trained-policy performance. Real multi-task
training updates also passed with the optimized kernels.

The vision profile measured only 0.26 s of a 16.9 s update, so vision caching
offers little saving here. An experimental supervised-position-only loss matched
synthetic loss/gradient checks but did not improve real training speed; it is
disabled. An audit found that sampling repetitive interior frames could halve
example count, but this has not been enabled. The full dataset remains intact.

## Time budget and launch

At approximately 6 s/update, the provisional full-data estimates are:

- **Five nominal passes:** approximately 12.5–17 hours of training compute.
- **Two nominal passes initially:** approximately 5–7 hours of training compute.

These assume approximately 7,500–10,000 updates for five passes; the exact count
depends on full conversion coverage. Preparation, validation, saving, first-use
compilation and final evaluation add wall time. Fewer passes reduce work, but
their policy quality is not guaranteed: evaluate checkpoints before extending.

[fast-training-command.txt](../commands/fast-training-command.txt) provides the two-pass
initial option; [command.txt](../commands/command.txt) retains the five-pass option. Both use
`train-fast --microbatch 32`, predicting one 1 cm / 5-degree action at a time.
The exporter keeps batch 16 as a compatible default; explicit launch overrides
are recorded with kernel versions in `training-launches/`.

Resume uses the same explicit checkpoint protocol from [RUNNING.md](RUNNING.md),
replacing `train` with `train-fast`. Changing the total pass count recalculates the
cosine/warm-up schedule and streamed replay is not guaranteed bit-identical.
Full training is not automatically launched by any setup or benchmark.

## Evidence and sources

See [fast-kernel benchmark](../evidence/fast-kernel-benchmark.json),
[Gated DeltaNet checks](../evidence/fla-numerical-check.json),
[convolution checks](../evidence/causal-numerical-check.json), and
[build lock](../evidence/fast-kernel-build-lock.json).
The original [FLA installation instructions](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
and [causal-conv1d repository](https://github.com/Dao-AILab/causal-conv1d)
describe these kernel packages.
