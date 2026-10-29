# Rotation-loop diagnostic

Checkpoint 3464 was queried on 48 held-out expert frames: all four validation episodes in each of alphabet soup (control), cream cheese, BBQ sauce and tomato sauce. Each episode contributed its first frame, first translation frame and pre-GRASP frame. Images, instruction, action history and expected label were taken from the exact frozen validation export. No actions were executed. The already-running evaluation server was used without changing its checkpoint or settings.

| Task | Exact start label | Exact first-translation label | Exact pre-GRASP label |
| --- | --- | --- | --- |
| Alphabet soup | 1/4 | 1/4 | 4/4 |
| Cream cheese | 2/4 | 1/4 | 4/4 |
| BBQ sauce | 1/4 | 3/4 | 3/4 |
| Tomato sauce | 4/4 | 3/4 | 2/4 |

Cream cheese predicts RT_YAW_CW at all four expert starts, even though two expect pitch actions. At three of four first-translation frames it still predicts a rotation rather than the demonstrated MV_FWD. However it predicts GRASP correctly at all four prepared grasp poses.

Tomato sauce predicts RT_YAW_CW at all four starts, but this is the demonstrated action in all four: its initial yaw choice is justified by these examples. On expert states after the orientation prefix, it switches to MV_FWD correctly in three of four episodes. Thus its live all-yaw behavior is not an inability to predict translation for that task at all. The rotation loop is more consistent with weak closed-loop transition/recovery from its own observations than universally wrong initial labels. This is a hypothesis, not a causal proof.

Across the four tasks it predicts the first translation label in 8/16 cases and GRASP in 13/16. Exact-match disagreement does not prove the alternative action cannot succeed; corrective rotations can differ. This deliberately selected small diagnostic is neither general action accuracy nor success rate.

Next useful isolation: compare expert and model rotation prefixes from the same initial state, including actual wrist orientation and camera frames at the moment translation should begin. This distinguishes executor/observation drift from an incorrectly learned stopping condition. Longer training remains worth testing, but this diagnostic alone does not establish it as the fix.

Detailed predictions and image paths: [evidence/rotation-prediction-predictions.json](../evidence/rotation-prediction-predictions.json). Summary: [evidence/rotation-prediction-summary.json](../evidence/rotation-prediction-summary.json).
