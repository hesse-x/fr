"""Fixed production compiler settings; no experimental backend switches."""
from pathlib import Path


def compiler_flags():
    if 'avx2' not in Path('/proc/cpuinfo').read_text().split():
        raise RuntimeError('channel native solver requires an AVX2 CPU')
    return ['-O3', '-std=c++17', '-march=native', '-fopenmp', '-ffp-contract=fast',
            '-fno-math-errno', '-shared', '-fPIC']


def header_bytes():
    return b''.join(p.read_bytes() for p in sorted(Path(__file__).with_name('native').glob('*.hpp')))


def locality_options():
    return dict(face_local=True, local_streaming_stores=True)
