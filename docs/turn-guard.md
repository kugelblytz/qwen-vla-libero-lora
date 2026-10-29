# Turn guard — diagnostic evaluation option

> **Status:** Implemented, not yet run. No results recorded.
> **Protocol:** Diagnostic. Not part of the frozen benchmark contract.

## Why

In the checkpoint-3464 native evaluation, 24/100 attempts issued RT_YAW_CW for
every one of their 400 commands (all cream-cheese and tomato-sauce attempts, three
BBQ sauce, one butter). The [pose analysis](../evidence/yaw-loop-poses.json) found
that each of these drove the arm away from its position after 26–27 tokens, 124–131
degrees of yaw, as did two attempts whose initial yaw run ended later (26 in total).
After that the controller can no longer hold the TCP and the attempt cannot recover.
According to `scripts/contract.py`, expert demonstrations never exceed 70 degrees of
net yaw.

The guard asks whether these tasks succeed when the unrecoverable over-rotation is
prevented. It does not repair the learned stopping condition.

## Mechanism

- Each decision measures the net world-frame turn since the attempt's first pose
  (`turned_since_start`, `scripts/contract.py`).
- When net yaw reaches `--turn-guard-deg` (for example 90), the yaw token that would
  turn further is withheld from that decision. Turning back is always allowed.
- Withholding uses the same guided-decoding path as the existing MV_* stuck guard:
  the model chooses among the remaining tokens.
- `0` (default) disables it. Guard-free evaluations keep the original
  `frozen-evaluation.json` contents.

## Reporting rules

- Output directories are named `eval-<budget>-turnguard<deg>-<timestamp>`.
- `frozen-evaluation.json`, every `resolved-settings.json` and `summary.json` carry
  `protocol: diagnostic-turn-guard` (or `diagnostic`) and the guard angle.
- Per attempt, the summary records `turn_guard_decisions` and `max_abs_yaw_deg`.
- Report guarded scores **separately** from the 19/100 frozen result. Never merge,
  average or substitute them.

## Run

Same as the frozen evaluation, with one extra argument:

```bash
bash scripts/pipeline.sh evaluate --run "$DATASET" --adapter "$CHECKPOINT" --attempts 10 --budget native --turn-guard-deg 90
```

## Open questions

- Does guarded cream cheese / tomato sauce switch to translation, or loop on another
  token (for example RT_YAW_CCW back and forth)?
- Do guarded attempts reach GRASP at a usable orientation?
- Pitch loops (milk, RT_PITCH_BACK) are not covered by this guard.
