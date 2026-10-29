"""Download exact evaluation weights and record local integrity evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_REV = '15852e8c16360a2fea060d615a32b45270f8a8fc'
ADAPTER_REV = 'c1c94dfa0f5f1e8827acbc9698ba5f765f1605e5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--profile', choices=['2b-sim', '4b-real'], default='2b-sim')
    args = parser.parse_args()
    large = args.profile == '4b-real'
    base_repo = 'Qwen/Qwen3.5-4B' if large else 'Qwen/Qwen3.5-2B'
    base_rev = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' if large else BASE_REV
    folder = 'qwen3_5_4b' if large else 'qwen3_5_2b_sim'
    model_root = 'models-4b' if large else 'models'
    manifest = 'qwen-models-4b.json' if large else 'qwen-models.json'
    root = args.runtime.resolve() / model_root
    specs = [
        ('base', base_repo, base_rev, ['*.json', '*.safetensors', '*.jinja', 'tokenizer*', '*.txt']),
        ('adapter', 'showlab/Show-Harness-VLMs', ADAPTER_REV, [folder + '/*']),
    ]
    records = []
    for name, repo, revision, patterns in specs:
        dest = root / name
        snapshot_download(repo_id=repo, revision=revision, local_dir=dest, allow_patterns=patterns)
        files = []
        for path in sorted(dest.rglob('*')):
            if not path.is_file() or '.cache' in path.relative_to(dest).parts:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            files.append(
                {
                    'path': str(path.relative_to(dest)),
                    'bytes': path.stat().st_size,
                    'sha256': digest.hexdigest(),
                }
            )
        records.append({'name': name, 'repository': repo, 'revision': revision, 'files': files})
    base = root / 'base'
    adapter = root / 'adapter' / folder
    assert (base / 'config.json').is_file()
    index = base / 'model.safetensors.index.json'
    shards = (
        set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    )
    assert all((base / shard).is_file() for shard in shards), 'Missing base weight shard'
    assert (adapter / 'adapter_model.safetensors').is_file()
    config = json.loads((adapter / 'adapter_config.json').read_text())
    assert config['base_model_name_or_path'] == base_repo, config
    report = {
        'profile': args.profile,
        'status': 'downloaded',
        'model_loaded': False,
        'adapter_rank': config['r'],
        'sources': records,
    }
    (args.runtime / manifest).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Weights verified in {root}; model inference has not been run.')


if __name__ == '__main__':
    main()
