"""Optional isolated FLA/Triton environment; --causal builds the CUDA convolution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from contract import root_path, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--causal', action='store_true')
p.add_argument('--cuda-home', type=Path, help='Existing toolkit matching the base trainer CUDA version')
a = p.parse_args()
root = root_path()
runtime = root / '.workspace'
uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
base = runtime / 'lf-paper-venv/bin/python'
fast = runtime / 'lf-fast-venv'
if not (fast / 'pyvenv.cfg').exists():
    subprocess.run([uv, 'venv', '--python', str(base), str(fast)], check=True)
site = Path(
    subprocess.check_output(
        [str(fast / 'bin/python'), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'],
        text=True,
    ).strip()
)
base_site = subprocess.check_output(
    [str(base), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True
).strip()
(site / 'base-trainer.pth').write_text('import site; site.addsitedir(' + repr(base_site) + ')\n')
subprocess.run(
    [
        uv,
        'pip',
        'install',
        '--python',
        str(fast / 'bin/python'),
        '--no-deps',
        'triton==3.7.1',
        'fla-core==0.5.2',
        'flash-linear-attention==0.5.2',
    ],
    check=True,
)

if a.causal:
    work = runtime / 'causal-build-toolchain'
    work.mkdir(exist_ok=True)
    toolkit = a.cuda_home
    if toolkit is None and Path('/usr/local/cuda/bin/nvcc').exists():
        toolkit = Path('/usr/local/cuda')
    lock = json.loads((Path(__file__).parents[1] / 'evidence/fast-kernel-build-lock.json').read_text())
    if toolkit is None:
        assert platform.system() == 'Linux' and platform.machine() == 'x86_64', (
            'Provide --cuda-home on this platform'
        )
        cuda = subprocess.check_output(
            [str(base), '-c', 'import torch;print(torch.version.cuda)'], text=True
        ).strip()
        assert cuda == '12.9', 'Local toolkit lock is CUDA 12.9; provide --cuda-home for your CUDA version'
        toolkit = work / 'root/usr/local/cuda-12.9'
        for record in lock['toolchain_packages'].values():
            file = work / Path(record['Filename']).name
            if not file.exists():
                urllib.request.urlretrieve(lock['toolchain_repository'] + record['Filename'], file)
            assert hashlib.sha256(file.read_bytes()).hexdigest() == record['SHA256']
            subprocess.run(['dpkg-deb', '-x', str(file), str(work / 'root')], check=True)
    assert (toolkit / 'bin/nvcc').exists(), 'CUDA compiler not found'
    archive = work / 'causal-conv1d-v1.7.0.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve(lock['source_url'], archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == lock['source_sha256']
    with tarfile.open(archive) as source:
        source.extractall(work, filter='data')
    directory = work / 'causal-conv1d-1.7.0'
    setup = directory / 'setup.py'
    text = setup.read_text()
    anchor = '    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI'
    assert text.count(anchor) == 1
    addition = (
        '    # Local build: only the detected GPU architecture; kernel source unchanged.\n'
        '    if not HIP_BUILD:\n'
        '        major, minor = torch.cuda.get_device_capability()\n'
        '        cc_flag = ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]\n\n'
    )
    setup.write_text(text.replace(anchor, addition + anchor))
    includes = list((Path(base_site) / 'nvidia').glob('*/include'))
    env = dict(
        os.environ,
        CUDA_HOME=str(toolkit.resolve()),
        MAX_JOBS='2',
        OMP_NUM_THREADS='4',
        CAUSAL_CONV1D_FORCE_BUILD='TRUE',
        CPATH=':'.join(map(str, includes)) + ':' + os.environ.get('CPATH', ''),
    )
    subprocess.run(
        [
            uv,
            'pip',
            'install',
            '--python',
            str(fast / 'bin/python'),
            '--no-deps',
            '--no-build-isolation',
            str(directory),
        ],
        env=env,
        check=True,
    )
subprocess.run(
    [
        str(fast / 'bin/python'),
        '-c',
        'from transformers.models.qwen3_5 import modeling_qwen3_5 as m; '
        'assert m.chunk_gated_delta_rule is not None; '
        'print("Optimized Gated DeltaNet available; complete fast path:",m.is_fast_path_available)',
    ],
    check=True,
)
write(
    runtime / 'fast-kernel-environment.json',
    dict(
        base_site_packages=base_site,
        fast_environment=str(fast),
        triton='3.7.1',
        fla='0.5.2',
        causal_requested=a.causal,
        caveat='Isolated compiler override; inherits pinned trainer and Torch from the base environment.',
    ),
)
