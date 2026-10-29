# Interim rollout review — first 63 episodes

> Interim snapshot, kept as written. The full evaluation has since completed at 19/100; final counts are in [results.md](results.md).

Checkpoint 3464; native 400-command budget; frozen 1 cm translations and 5-degree rotations. Full evaluation remained in progress at the time. This review covers complete action traces, final camera views of all 63 episodes, representative intermediate views across all seven started tasks, and simulator reconstruction of all 63 final states. It is not a frame-by-frame review of every video.

| Task | LIBERO successes | GRASP issued | RELEASE issued | Entire episode RT_YAW_CW |
| --- | --- | --- | --- | --- |
| Alphabet soup | 3/10 | 9/10 | 1/10 | 0 |
| Cream cheese | 0/10 | 0/10 | 0/10 | 10 |
| Salad dressing | 3/10 | 10/10 | 0/10 | 0 |
| BBQ sauce | 0/10 | 6/10 | 0/10 | 3 |
| Ketchup | 5/10 | 9/10 | 0/10 | 0 |
| Tomato sauce | 0/10 | 0/10 | 0/10 | 10 |
| Butter (partial) | 0/3 | 2/3 | 0/3 | 0 |

## Observations

1. Failure is not limited to placing. Of 52 failures, 27 never issue GRASP. Twenty-three use RT_YAW_CW for all 400 commands: every cream-cheese and tomato-sauce attempt, plus three BBQ-sauce attempts. Those are systematic task-dependent policy failures from the initial observation, not a shortage of placement time.
2. Twenty-five failures issue GRASP; sampled frames confirm useful target grasp/lift on multiple tasks, followed by failed transport, overshoot, occasional loss of the object, or repeated movement. Issuing GRASP does not by itself establish a successful physical grasp. Ten failed final states still have both finger pads contacting the target.
3. LIBERO reports 11/63 successes (17.5%). Reconstructing final states reproduces every reported predicate result, but 10/11 successes still have both finger pads contacting the target and never issue RELEASE. Only alphabet-soup attempt 004 issues RELEASE and terminates without target finger contact. Thus report official predicate success separately from clean place-and-release behavior. Do not retroactively change the benchmark score or interrupt the ongoing run.
4. The action contract is consistent across conversion, training and evaluation: 1 cm translation, 5-degree rotation. The isolated grasp controls physically lift about 9–10 mm per upward command. The evidence does not support an unconverted 2 cm executor as the explanation.

## Conclusions and next checks

Local fine-tuning learned useful approach/grasp/transport behavior, but it is inconsistent across tasks and poor at phase transitions and recovery. The first-task conclusion that placement dominates does not generalize to cream cheese or tomato sauce. The native 400-decision limit is sufficient for several successes; extending it alone will not repair all-action yaw loops.

Longer training is a reasonable experiment, not a demonstrated fix. Before a broad extension, compare checkpoints 1732 and 3464 on held-out initial/approach frames for cream cheese and tomato sauce, and on placement/release frames across tasks. Then test five total passes against the same fixed rollout starts. If release remains weak, consider targeted sampling of complete local transition sequences rather than GRASP labels alone. The 1 cm representation dilutes rare transitions, but class imbalance and undertraining remain hypotheses, not established root causes.

There is no policy reward for release in the simulator predicate: evaluation stops as soon as success holds. A separate diagnostic can continue through release and verify stable placement without finger contact; that must be reported as an additional metric, without changing the frozen benchmark.

## Evidence

[evidence/rollout-review-63/summary.json](../evidence/rollout-review-63/summary.json) contains task/action counts. [Final simulator checks](../evidence/rollout-review-63/final-state-check.json) contain target finger contact and independently rechecked success. Contact sheets named taskXX.jpg show representative temporal views, and finals-taskXX.jpg show every completed final external-camera view. Full videos and action traces remain in full-eval-videos.
