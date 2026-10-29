# References and evaluation context

## Primary sources

- [Show-Harness: Just a VLM Agent Can Play Robots](https://arxiv.org/abs/2609.10522), Chen et al., Show Lab, September 2026.
- [Show-Harness repository](https://github.com/showlab/Show-Harness).
- [Released adapters](https://huggingface.co/showlab/Show-Harness-VLMs): `qwen3_5_2b_sim` is one Qwen3.5-2B LoRA trained jointly on ManiSkill and RoboLab, using prompt `v3`.
- [LIBERO integration](https://github.com/showlab/Show-Harness/pull/4), merged September 30, 2026: setup, semantic executor, camera contract, replay, and evaluation/data-generation scripts.
- [Standard LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): public task definitions, assets, and initial states.

## Why the target changed

The intended ManiSkill reproduction depended on `BlockPAP-v1`, loaded by Show-Harness from `real_franka.real2sim_env.pick_and_place` in an external RLinf checkout. Searches on 2026-10-04 across all eight public RLinf branches, the relevant path history, Show-Harness history/releases, GitHub code search, and released model/data files did not locate the implementation or scene-asset bundle. The environment is unavailable to this project; that does not prove it is private.

LIBERO provides an accessible alternative with an upstream Show-Harness runner. No publicly released LIBERO-trained Show-Harness adapter was found in the official checkpoint repository or Show Lab model listings at inspection.

## What this evaluation establishes

The milestone measures whether the released simulation adapter can operate through the LIBERO semantic-action loop and how it performs on all 10 standard `libero_spatial` tasks without additional training. Task success, action validity, latency, and failures describe transfer to this setup.

The paper's main experiments and sim-to-real evaluation concern physical robots. This LIBERO baseline does not reproduce those scores, validate sim-to-real transfer, or establish superiority over VLA baselines. Failure in LIBERO alone would not refute the original results.
