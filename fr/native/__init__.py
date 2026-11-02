"""Shared header-only native kernels and build-cache dependencies."""
from pathlib import Path


def header_bytes():
    """Include shared headers in extension cache keys after any kernel edit."""
    return b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.hpp')))
