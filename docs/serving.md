# Qwen serving environment

From the project root, install `uv` if needed (`python -m pip install --user uv`), ensure its executable is on `PATH`, and run:

```bash
bash scripts/sim/setup_qwen.sh
```

The script creates `.workspace/qwen-venv` with managed Python 3.11, separate from LIBERO. It installs the pinned Qwen3.5-compatible vLLM stack, checks dependency consistency, executes PyTorch and vLLM CUDA kernels, and imports the Qwen3.5 architecture. Results go to `.workspace/qwen-environment.json`. Model weights are not downloaded and inference is not tested by this command.

## Portability

Supports Linux x86_64 and aarch64 with NVIDIA GPUs supported by vLLM. Paths are derived from the repository location. The installed CUDA toolkit is not used: prebuilt wheels bring their runtime libraries. The NVIDIA driver and GPU capabilities still determine compatibility.

The script selects CUDA 12.9 wheels for driver branches 525–579 and CUDA 13.0 wheels for 580+. CUDA 12 minor-version compatibility allows older drivers, but some PTX/JIT paths may require a newer driver; passing the environment check does not prove every inference kernel works. See [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) and [NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Optional overrides:

```bash
QWEN_CUDA_PROFILE=cu129 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/sim/setup_qwen.sh
```

`QWEN_CUDA_PROFILE` accepts `auto`, `cu129`, or `cu130`. `VLA_WORKSPACE`, `QWEN_VENV`, `UV_BIN`, and `UV_CACHE_DIR` allow alternate locations. Normal `CUDA_VISIBLE_DEVICES` selection is preserved. No GPU name, device count, VRAM size, system toolkit path, or home directory is hardcoded. Runtime serving memory limits must be chosen for the actual GPU when launching the model.

The checked-in CUDA 12.9/x86_64 lock records the verified installation. Other profiles resolve the same pinned top-level versions once and retain a full lock in the runtime directory; keep that lock with evaluation evidence. Only CUDA 12.9/x86_64 has been tested here. Rerunning syncs the chosen lock and repeats the checks.

## Verified reference

Python 3.11.17, vLLM 0.24.0+cu129, PyTorch 2.11.0+cu129, Transformers 5.12.1, huggingface-hub 1.17.0. Tested on driver 535.247.01 and an H100 MIG 2g.20gb device. See [environment evidence](../evidence/setup/qwen-environment.json).

## Pinned model and adapter

Run from the project root after installing the environment:

```bash
.workspace/qwen-venv/bin/python \
  scripts/sim/download_qwen.py \
  --runtime .workspace
bash scripts/sim/serve_qwen.sh
```

The downloader uses the exact base and adapter commits in [setup.md](setup.md), checks required weight files and adapter compatibility, and records SHA-256 hashes in `.workspace/qwen-models.json`. Downloads can be resumed by rerunning the command. Reference download evidence is in [qwen-models.json](../evidence/setup/qwen-models.json). Weights stay outside Git.

To avoid requiring a system `nvcc`, the launcher defaults to the PyTorch sampler, FlashAttention, and Triton GDN prefill. These preserve the trained model and adapter; they select execution backends. Override `VLLM_USE_FLASHINFER_SAMPLER`, `ATTENTION_BACKEND`, or `GDN_PREFILL_BACKEND` only when the machine supports the corresponding dependencies.

The launcher verifies the upstream checkout revision, serves local weights offline with `qwen3_5_nothink.jinja`, registers `qwen3_5_2b_showharness_sim`, and listens on localhost port 8000. It defaults to one concurrent sequence, an 8192-token context, eager execution, and 80% GPU memory utilization. Override `GPU_UTIL`, `MAX_LEN`, `PORT`, `TP`, `CUDA_VISIBLE_DEVICES`, or the runtime location as needed. `DRY_RUN=1` prints the command. Use prompt `v3` in the client.

Server startup, internal warmup, the health endpoint, and registration of both served model names passed on the reference machine. See [serving evidence](../evidence/setup/qwen-serving-check.json). The server was stopped after validation to free GPU memory. Subsequent inference and a LIBERO diagnostic episode passed pipeline verification; see pipeline evidence. The diagnostic reached 200 actions without task success.

## Start the full evaluation yourself

From the project root:

```bash
nohup bash scripts/sim/run_full_eval.sh > full-eval.log 2>&1 &
tail -f full-eval.log
```

The launcher starts its own model server, waits for health, then runs 30 tasks × 10 attempts (initial-state indices 0–9), stopping the server when the batch ends. The port must be free. `Ctrl+C` while following the log only stops `tail`; the background evaluation continues. Save the process ID printed by the shell if you want to stop the evaluation with `kill -TERM <pid>`.

Results go to a new timestamped directory under `artifacts/evaluation/`, printed in the log. Each `<suite>/<task>/attempt-NNN/` contains request/response logs, frames, states, episode outcome, and `trajectory.mp4` (agent view left, wrist view right, four decisions per second plus the final observation). Each completed attempt must produce its video before continuing. A simulator failure before any image is available is retained as an infrastructure error; no video can be created from missing observations. Infrastructure errors stop the batch without replacing attempts. Earlier results remain intact.

The root contains `batch.log`, `server.log`, pinned setup evidence, installed package versions, the server command, `exit-code.txt`, and—after all 300 attempts—`summary.json`. An exit code of 0 means the batch completed, regardless of manipulation success rate.

`EVAL_OUT` sets a custom **new** output directory. `PORT` and the serving/rendering overrides above are respected. Use `bash scripts/sim/run_full_eval.sh --dry-run` to print all 300 commands and video destinations without starting serving or rollouts. `ffmpeg` is required and available on the current machine. The restarted launcher uses boundary rendering and snapshots its adapter source with each batch. The short model test completed 20 decisions in 10.4 seconds; full 200-decision attempts are expected to be around two minutes plus process/reset overhead, subject to task-dependent control-step counts. See rendering verification.

### Optional 4B comparison

Use `scripts/run_full_eval_4b.sh` for the same 30 tasks × 10 initial states,
200-decision limit, seed, v3 prompt, boundary rendering, and video protocol.
Only the base model and adapter change: Qwen3.5-4B with the released
`qwen3_5_4b` real-robot adapter. This comparison changes training corpus as well
as model size; it is not a controlled scaling experiment or a LIBERO-trained policy.

Weights are separate in `.workspace/models-4b`; the pinned base revision is
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and the adapter uses the original
`c1c94dfa0f5f1e8827acbc9698ba5f765f1605e5` release. Integrity hashes are recorded
in `.workspace/qwen-models-4b.json` and copied into each batch.

To prepare weights on another machine, use its Qwen environment:

```bash
.workspace/qwen-venv/bin/python scripts/sim/download_qwen.py --runtime .workspace --profile 4b-real
```

Launch using `command-4b.txt`. Results go into a fresh `run-4b-<timestamp>`
directory; logs go into `full-eval-4b.log`. Stop any existing model server first,
since the launcher requires a free port and the comparison uses the same GPU.
The original default launcher still selects the 2B simulation adapter.

Validated on this machine: the 4B BF16 model and rank-64 adapter loaded,
a saved two-camera LIBERO request returned a valid action token with HTTP 200,
and both launcher dry runs produced 300 attempts and 300 video destinations.
No comparison rollouts were run during setup. Evidence: `artifacts/setup/qwen-4b-serving-check.json`.
