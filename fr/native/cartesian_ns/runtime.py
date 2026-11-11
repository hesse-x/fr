"""Process-local CPU allocation policy; all numerical work stays in Torch."""
import ctypes
import os


def configure_cpu_memory_reuse(enabled=True):
    """Keep glibc's large temporary buffers available between Torch calls.

    Linux/glibc mallopt constants: M_TRIM_THRESHOLD=-1, M_MMAP_THRESHOLD=-3.
    This changes allocation policy for this process, never the system allocator.
    Memory is retained until reused or process exit; disabling avoids the calls.
    """
    result = dict(enabled=False, allocator='unchanged')
    if not enabled:
        return result
    try:
        libc = ctypes.CDLL(None)
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        version = libc.gnu_get_libc_version().decode('ascii')
        mallopt = libc.mallopt
    except (AttributeError, OSError):
        result['reason'] = 'glibc mallopt unavailable'
        return result
    threshold = int(os.environ.get('MALLOC_MMAP_THRESHOLD_', 1024**3))
    trim = int(os.environ.get('MALLOC_TRIM_THRESHOLD_', -1))
    mallopt.argtypes = (ctypes.c_int, ctypes.c_int)
    mallopt.restype = ctypes.c_int
    if mallopt(-3, threshold) != 1 or mallopt(-1, trim) != 1:
        raise RuntimeError('glibc rejected the CPU memory reuse policy')
    return dict(enabled=True, allocator='glibc', version=version,
                mmap_threshold_bytes=threshold, trim_threshold_bytes=trim)
