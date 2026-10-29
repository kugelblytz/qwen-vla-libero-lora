# Evidence

Compact JSON summaries behind the numbers in [`docs/`](../docs/). Each results document links the files it
uses. Large outputs (videos, decision and pose logs, checkpoints) stayed on the training server.

| Folder | Contents |
|---|---|
| `.` | v5 run: dataset manifest, training summary, frozen evaluation, failure analyses |
| `v6/` | 4B v6 run: training summary, changepoint sweep, frozen evaluation, v5-vs-v6 metrics |
| `v6-2b/` | 2B v6 run: the same for Qwen3.5-2B, plus 4B-vs-2B metrics |
| `single-task/` | single-task 4B run (epoch 1 vs epoch 40) |
| `setup/` | environment, rendering and serving checks, pinned task list |

Files are kept exactly as recorded. Absolute paths inside them (`/home/jovyan/…/.milestone01/…`) refer to
the original server layout; in this repository that directory is `.workspace/`.
