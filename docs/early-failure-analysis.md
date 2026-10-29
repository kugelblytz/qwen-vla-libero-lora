# Early rollout failure analysis

Snapshot: first seven alphabet-soup attempts, checkpoint 3464, native 400-command budget. Two succeeded and five failed. (Interim snapshot; the full evaluation has since completed at 19/100, see [results.md](results.md).)

Four failures (initial-state indices 0, 1, 2, 3) issued GRASP and visibly lifted the correct target. All kept holding it without RELEASE and reached 400 commands. Three ended in repeated MV_UP commands; another ended mostly in MV_BACK commands. Thus these are downstream transport/placement or recovery failures, not empty grasps. The fifth failure (index 5) never issued GRASP and ended repeating MV_FWD: an approach/phase-transition failure.

Success index 4 transported, descended and issued RELEASE. Index 6 satisfied the LIBERO success predicate without issuing RELEASE, so benchmark success does not always imply a clean completed place-and-release sequence.

Evidence: paired frames in evidence/early-placement-comparison.jpg and evidence/early-failed-grasps-lifted.jpg, action-count snapshot in evidence/early-failure-analysis.json, complete videos in full-eval-videos/. The held-out pre-grasp diagnostic also passed 3/3.

These observations establish failure stages on one task. They do not establish the training root cause. Insufficient repetition, sparse critical action transitions, and recovery from states outside expert trajectories are hypotheses requiring follow-up comparisons. Later tasks may fail differently.
