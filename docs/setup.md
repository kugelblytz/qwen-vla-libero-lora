# Setup

Use standard LIBERO with the released Show-Harness simulation adapter; no further training or policy tuning.

## Reference setup

- `scripts/serve_vlm.sh` — vLLM serving with family `qwen3_5` and adapter name `qwen3_5_2b_showharness_sim`.
- `scripts/setup.sh libero <checkout>` — isolated LIBERO environment.
- `scripts/run_libero_mvtoken.py` — autonomous semantic-action runner.
- `configs/robot_libero.yaml` — agent/wrist camera transforms.
- Standard [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) checkout — task definitions, assets, and provided initial states. No RLinf checkout is required.

Follow the pinned upstream [simulator guide](https://github.com/showlab/Show-Harness/blob/e9867e971c3a8c5da352e151006cac859923e85e/docs/simulators.md). Select all 10 `libero_spatial` task names explicitly with `--tasks`; the runner's default discovers task names from a locally generated dataset that we do not have.

## Reference freeze — 2026-10-04

Show-Harness and LIBERO are checked out at these revisions. Task loading, rendering, action controls, model serving, and one autonomous diagnostic loop have passed. The diagnostic did not succeed at manipulation; suite evaluation remains pending.

| Component | Source | Pinned revision |
|---|---|---|
| Show-Harness | [showlab/Show-Harness](https://github.com/showlab/Show-Harness/tree/e9867e971c3a8c5da352e151006cac859923e85e) | `e9867e971c3a8c5da352e151006cac859923e85e` |
| Base model | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Simulation adapter | [showlab/Show-Harness-VLMs, qwen3_5_2b_sim](https://huggingface.co/showlab/Show-Harness-VLMs/tree/c1c94dfa0f5f1e8827acbc9698ba5f765f1605e5/qwen3_5_2b_sim) | `c1c94dfa0f5f1e8827acbc9698ba5f765f1605e5` |
| LIBERO | [Standard upstream checkout](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |

The adapter metadata does not pin a base-model revision; the selected base revision above is our evaluation reference. All released adapters use prompt `v3`; explicitly pass `--version v3` because the LIBERO runner defaults to `v5`. No publicly released LIBERO-trained Show-Harness adapter was found.

**Available machine:** Ubuntu 22.04.4, NVIDIA driver 535.247.01, H100 MIG `2g.20gb` with 19.62 GiB usable VRAM (19.49 GiB free at inspection), 64 GiB container RAM, 24-core CPU quota, and approximately 247 GB free workspace storage.

**Deployment:** start with serving and simulation on this machine in separate environments. Confirm headless MuJoCo rendering and combined memory use; split machines if needed. The existing PyTorch environment is diagnostic only.

Upstream specifies Python 3.10 / PyTorch 2.6.0+cu124 for LIBERO and vLLM 0.24.0 / Transformers 5.12.1 / huggingface-hub 1.17.0 for serving. The LIBERO environment is installed with Python 3.10.22, PyTorch 2.6.0+cu124, MuJoCo 3.3.2, and NumPy 1.26.4. The separate Qwen serving environment is installed and passes CUDA computation, vLLM native-kernel, and Qwen3.5 import checks. Pinned model weights and the simulation adapter have been downloaded, with file hashes recorded. Server startup, internal warmup, health, and adapter registration have passed; client requests and one autonomous LIBERO smoke test have now passed pipeline verification (evidence); the diagnostic reached the action limit without task success. See [portable serving setup](serving.md) for the installation command, driver selection, and dependency lock.

## Completed setup checks

The upstream setup script installed an isolated environment under `.workspace/LIBERO/.venv`. Checkouts, package caches, and locally extracted system libraries stay under `.workspace/`, excluded from Git through `.git/info/exclude`.

All 10 standard `libero_spatial` task definitions and provided initial-state files loaded. The first task in order 0 reset to initial-state index 0 and rendered both views at 256×256 through the upstream backend. Saved images were inspected. This checks environment availability, not autonomous control or task success; translation and gripper validation have since passed (measurements).

The container lacks usable EGL/OSMesa libraries by default. Rendering passed with `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, and Ubuntu OSMesa libraries extracted locally, without modifying the system installation. This uses CPU software rendering; GPU rendering remains untested; the model and software-rendered simulator ran concurrently during the diagnostic. The training-demonstration dataset is not needed for task reset or evaluation.

Evidence:

- [reference.json](../evidence/setup/reference.json) — exact 10-task list, code revisions, BDDL and initial-state file hashes.
- [libero-packages.txt](../evidence/setup/libero-packages.txt) — installed Python dependency versions.
- [render-libraries.json](../evidence/setup/render-libraries.json) — exact Ubuntu packages, download URLs, and checksums.
- [render-check.json](../evidence/setup/render-check.json) — scene, fixture poses, camera settings, and image checks.
- [Agent view](../evidence/setup/agentview.png) and [wrist view](../evidence/setup/wrist.png) — transformed input frames; `scene_state.npy` beside them preserves the rendered state.

### Reproduce steps 1–2

Run from the project root. Clone the two source repositories into `.workspace/Show-Harness` and `.workspace/LIBERO`, and check out the table's commits. Install `uv` if missing, then run:

```bash
cd .workspace/Show-Harness
PATH="$HOME/.local/bin:$PATH" bash scripts/setup.sh libero ../LIBERO
```

For this container, download the packages listed in `render-libraries.json` into `.workspace/system-libs/packages`, verify their SHA-256 values, and extract each with `dpkg-deb -x <package> .workspace/system-libs/root`. Return to the project root and run:

```bash
export LD_LIBRARY_PATH="$PWD/.workspace/system-libs/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
.workspace/LIBERO/.venv/bin/python \
  scripts/sim/check_libero_setup.py \
  --show-harness .workspace/Show-Harness --libero .workspace/LIBERO \
  --out evidence/setup
```

The upstream installer leaves some dependencies unpinned; compare installations against `libero-packages.txt` before treating them as the same setup. The check script records the actual revisions and overwrites its output directory on rerun; use a different `--out` to preserve earlier evidence.

## Essential checks and reproduction record

1. Reset the first selected task and inspect both views using the upstream camera transforms and preprocessing.
2. Verify translation directions, approximately 2 cm movement, and gripper execution before querying the model.
3. Run one autonomous diagnostic episode, preserving the request, action, and resulting observation sequence. Task success is not required for this smoke test.

Preserve code/model revisions, installed dependency versions, OS/GPU details, exact launch commands, prompt/chat template, and all resolved runner settings. Record parsing, action constraints, rotation support, stuck-guard behavior, and timeout/retry handling. Keep the released adapter's interface; do not silently enable an untrained action vocabulary. Store initial-state indices and fixture poses with rollouts so replay reconstructs the scene.

Apply only the protocol settings in [evaluation.md](evaluation-protocol.md); detailed configuration belongs in the saved artifact rather than repeated documentation.
