import argparse
from pathlib import Path
from contract import import_sim, root_path, write

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--tasks', default='0,1,2,3,4,5,6,7,8,9')
a = p.parse_args()
root = root_path()
import_sim(root)
from libero.libero import benchmark, get_libero_path

suite = benchmark.get_benchmark_dict()['libero_object'](task_order_index=0)
indices = [int(i) for i in a.tasks.split(',')]
assert len(set(indices)) == len(indices) and all(0 <= i < 10 for i in indices)
tasks = []
for i in indices:
    task = suite.get_task(i)
    tasks.append(
        dict(
            index=i,
            name=Path(task.bddl_file).stem,
            bddl=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
        )
    )
write(a.run / 'tasks.json', tasks)
