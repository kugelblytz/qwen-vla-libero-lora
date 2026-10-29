# Single-task follow-up results — 9 October 2026

Qwen3.5-4B with the retained **epoch-40 / step-2040** adapter achieved
**6/10 successes** on cream-cheese pick-and-place (`libero_object`, task index 1).
The epoch-1 adapter from the same training run achieved 0/10 on the same ten
benchmark initial states. Both evaluations used v5, 2 cm translations,
10-degree rotations, 25 gripper control steps, a 200-action limit, a 16-token
response budget and boundary rendering. The model was served in BF16 through
vLLM. These initial states are not claimed to be disjoint from training states.

## Epoch-40 outcomes

| Attempt | Result | Actions | Termination | GRASP | RELEASE |
| --- | --- | ---: | --- | ---: | ---: |
| attempt-000 | Success | 66 | success | 1 | 1 |
| attempt-001 | Failure | 63 | done | 1 | 0 |
| attempt-002 | Failure | 66 | done | 2 | 0 |
| attempt-003 | Success | 67 | success | 1 | 1 |
| attempt-004 | Failure | 62 | done | 1 | 0 |
| attempt-005 | Success | 66 | success | 1 | 1 |
| attempt-006 | Success | 62 | success | 1 | 1 |
| attempt-007 | Success | 64 | success | 1 | 1 |
| attempt-008 | Failure | 68 | done | 1 | 0 |
| attempt-009 | Success | 63 | success | 1 | 1 |

All ten attempts completed and have two-camera videos. Four failed attempts
issued DONE before LIBERO's success predicate was satisfied. They issued GRASP
but no RELEASE; this alone does not identify whether grasping or placement failed.

## Demonstration replay

The same ten source demonstrations were converted and cumulatively replayed at
2 cm and 1 cm. Both replay conditions achieved 9/10 successes; the same episode
(`validation:1`) failed. Total actions increased from 702 to 1,333 (1.90 times).
The regenerated 1 cm source conversion itself also failed that episode.

This tests expert-command execution rather than learned-policy performance.
Smaller steps also change lattice anchoring and potentially the follower's path;
planner tolerances were unchanged and the token budget increased from 300 to 600.
No success improvement or benefit from a trained 1 cm policy has been established.

## Evidence

Compact evidence preserved in Git:

- [Epoch-40 rollout summary](../evidence/single-task/epoch40-summary.json).
- [Epoch-1 rollout summary](../evidence/single-task/epoch1-summary.json).
- [Evaluated checkpoint identity](../evidence/single-task/epoch40-adapter.json) and
  [adapter SHA-256](../evidence/single-task/epoch40-adapter-sha256.txt).
- [Paired step-size results](../evidence/single-task/step-comparison.json).

Large logs, images and videos are retained locally, outside Git:

- Epoch 40: `epoch40-eval-videos/`, linked to
  `.workspace/libero-4b-v5-eval-20261009T135917-178992`.
- Epoch 1: `eval-videos/`, linked to
  `.workspace/libero-4b-v5-eval-20261009T100100-135777`.
- Expert replay: `replay-results/` and `step-comparison-results/`.

These results support local single-task fine-tuning and show that selecting by
aggregate validation loss can miss a more useful controller. Ten attempts on
one task do not establish broader LIBERO reliability or reproduce the paper.
