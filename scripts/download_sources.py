"""Download pinned original LIBERO Object HDF5 demonstrations, without credentials."""

import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from contract import digest, root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
root = root_path()
source = json.loads((root / '.workspace/libero-demo-source.json').read_text())
tasks = json.loads((a.run / 'tasks.json').read_text())
files = []
for task in tasks:
    filename = f"libero_object/{task['name']}_demo.hdf5"
    path = Path(
        hf_hub_download(
            source['repo'],
            filename,
            repo_type='dataset',
            revision=source['revision'],
            local_dir=root / '.workspace/libero-demos',
        )
    )
    files.append(dict(task=task['index'], name=task['name'], path=str(path), sha256=digest(path)))
    print('Ready:', filename, flush=True)
write(
    a.run / 'sources.json',
    dict(
        repo=source['repo'],
        revision=source['revision'],
        files=files,
        selection='Original 50-episode/task source; raw replay success checked before conversion. Exact OpenVLA cleaned episode selection has not been established.',
    ),
)
