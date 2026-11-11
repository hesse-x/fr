"""Compiler settings for the Cartesian NS preset; separate boundary sweeps by default."""
from pathlib import Path
from fr.native import aten_compile_flags, header_bytes as shared_header_bytes


def compiler_flags(boundary_mode='separate'):
    if 'avx2' not in Path('/proc/cpuinfo').read_text().split():
        raise RuntimeError('Cartesian NS native solver requires an AVX2 CPU')
    if boundary_mode not in ('separate', 'fused'):
        raise ValueError('boundary_mode must be separate or fused')
    mode = str(int(boundary_mode == 'separate'))
    return ['-DFR_NS_SPLIT_BOUNDARY='+mode, '-O3', '-std=c++17', '-march=native', '-fopenmp', '-ffp-contract=fast',
            '-fno-math-errno', '-shared', '-fPIC'] + aten_compile_flags()


def header_bytes():
    return shared_header_bytes() + b''.join(p.read_bytes() for p in sorted(Path(__file__).with_name('kernels').glob('*.hpp')))


def locality_options():
    return dict(face_local=True, local_streaming_stores=True)
