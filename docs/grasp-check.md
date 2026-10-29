# Focused grasp check

Checkpoint 3464 passed all three held-out pre-grasp tests. Expert controls also passed all three.

| Target | Policy | Expert | Policy maximum lift |
| --- | --- | --- | --- |
| Alphabet soup | Pass | Pass | 4.86 cm |
| Cream cheese | Pass | Pass | 5.54 cm |
| Chocolate pudding | Pass | Pass | 5.55 cm |

Pass required target elevation of at least 3 cm with contact at both finger pads for three consecutive action boundaries. Each episode was capped at 24 decisions. The policy selected GRASP followed by MV_UP commands, matching the expert prefixes.

These states came from validation episodes and were restored immediately before the converted expert GRASP action. No approach was predicted. This checks local grasp and lift behavior, not full-task success or benchmark performance. Only three states were tested.

Videos and detailed measurements: grasp-check-videos. Each task has an expert and model directory containing trajectory.mp4; agent camera is left, wrist camera is right. Summary and frozen adapter selection are in evidence/grasp-check.json and evidence/grasp-check-selection.json.
