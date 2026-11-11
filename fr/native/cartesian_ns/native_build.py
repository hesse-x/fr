"""Compiler settings for the Cartesian NS preset; separate boundary sweeps by default."""
from pathlib import Path
from fr.native import aten_compile_flags, header_bytes as shared_header_bytes


def compiler_flags(boundary_mode='separate', viscous_config=None, sgs=None, interface_flux='rusanov', anti_alias=(), quadrature_order=None):
    if 'avx2' not in Path('/proc/cpuinfo').read_text().split():
        raise RuntimeError('Cartesian NS native solver requires an AVX2 CPU')
    if boundary_mode not in ('separate', 'fused'):
        raise ValueError('boundary_mode must be separate or fused')
    mode = str(int(boundary_mode == 'separate'))
    if interface_flux not in ('rusanov', 'roe'):
        raise ValueError('interface_flux must be rusanov or roe')
    viscous_flags = ['-DFR_NS_ROE=1'] if interface_flux == 'roe' else []
    if anti_alias:
        nq = 5 if quadrature_order is None else quadrature_order
        if not isinstance(nq, int) or nq < 4:
            raise ValueError('native-p3 quadrature order must be an integer >=4')
        viscous_flags += ['-DFR_NS_NQ='+str(nq)]
        if 'flux' in anti_alias:
            viscous_flags += ['-DFR_NS_VOLUME_AA=1']
        if 'surf-flux' in anti_alias:
            viscous_flags += ['-DFR_NS_SURFACE_AA=1']
    if viscous_config is not None and viscous_config.scheme == 'br1-penalty':
        viscous_flags += ['-DFR_NS_BR1=1', '-DFR_NS_PENALTY='+repr(float(viscous_config.penalty))]
    if sgs is not None:
        from fr.les import Vreman, Smagorinsky
        if type(sgs) not in (Vreman, Smagorinsky):
            raise ValueError('native-p3 SGS requires Vreman or Smagorinsky')
        if viscous_config is None or viscous_config.scheme != 'ldg':
            raise ValueError('native-p3 SGS requires LDG')
        viscous_flags += ['-DFR_NS_VREMAN=1' if type(sgs) is Vreman else '-DFR_NS_SMAGORINSKY=1']
    return viscous_flags+['-DFR_NS_SPLIT_BOUNDARY='+mode, '-O3', '-std=c++17', '-march=native', '-fopenmp', '-ffp-contract=fast',
            '-fno-math-errno', '-shared', '-fPIC'] + aten_compile_flags()


def header_bytes():
    return shared_header_bytes() + b''.join(p.read_bytes() for p in sorted(Path(__file__).with_name('kernels').glob('*.hpp')))


def locality_options():
    return dict(face_local=True, local_streaming_stores=True)
