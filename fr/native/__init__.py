"""Shared header-only native kernels and build-cache dependencies."""
from pathlib import Path


def header_bytes():
    """Include shared headers in extension cache keys after any kernel edit."""
    return b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.hpp')))


def aten_compile_flags():
    """Compile/link C ABI kernels against the active PyTorch ATen runtime."""
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    flags = [f'-I{path}' for path in include_paths()]
    flags += [f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}']
    for path in library_paths():
        flags += [f'-L{path}', f'-Wl,-rpath,{path}']
    # These flags precede the translation unit in the existing compiler commands.
    flags += ['-Wl,--no-as-needed', '-ltorch_cpu', '-lc10', '-Wl,--as-needed']
    return flags
