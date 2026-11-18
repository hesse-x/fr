"""Native adapter for the migrated HiFiLES concentration/modal filter."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import time
import torch
from .native_build import compiler_flags, header_bytes

_LIB = None


def load_library():
    global _LIB
    if _LIB is not None:
        return _LIB
    source = Path(__file__).with_name('kernels')/'modal_filter.cpp'
    flags = compiler_flags()
    key = hashlib.sha256(source.read_bytes()+header_bytes()+torch.__version__.encode()
                         +' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-cartesian-ns-native')/key
    build.mkdir(parents=True, exist_ok=True)
    target = build/'modal_filter.so'
    if not target.exists():
        command = ['g++', *flags, str(source), '-o', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        (build/'command.txt').write_text(' '.join(command)+'\n')
        (build/'build.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    _LIB = ctypes.CDLL(str(target))
    p = ctypes.c_void_p
    _LIB.mapped_hex_modal_filter.argtypes = [ctypes.c_int64]+[p]*5+[ctypes.c_int,
        ctypes.c_double, ctypes.c_int]+[p]*3
    _LIB.mapped_hex_modal_filter.restype = ctypes.c_int
    return _LIB


class NativeCartesianModalFilter:
    def __init__(self, model):
        if model.s.p != 3 or model.s.z.device.type != 'cpu':
            raise ValueError('native modal filter requires CPU p=3')
        self.model = model
        self.filter = model.modal_filter
        geometry = self.filter.geometry
        matrices = self.filter.closure.matrices(geometry)
        self.matrix, self.sensor = [torch.from_numpy(v) for v in matrices]
        self.J = torch.from_numpy(geometry.J).contiguous()
        self.weights = torch.from_numpy(geometry.weights).contiguous()
        self.count = self.J.shape[0]
        self.cell_sensor = torch.zeros(self.count, dtype=torch.float64)
        self.hits = torch.zeros(self.count, dtype=torch.int64)
        self.statistics = torch.zeros(4, dtype=torch.float64)
        self.library = load_library()
        self.failed_state = None
        self.calls = self.changed = 0
        self.seconds = self.maximum_sensor = self.maximum_mean_error = 0.

    def apply(self, state):
        """Filter a CPU FP64 SoA state in place; never clip density or pressure."""
        if (tuple(state.shape) != self.model.s.shape+(5,) or state.dtype != torch.float64
                or state.device.type != 'cpu' or not state.movedim(-1, 0).is_contiguous()):
            raise ValueError('expected matching CPU FP64 SoA state')
        started = time.perf_counter()
        status = self.library.mapped_hex_modal_filter(self.count, state.data_ptr(),
            self.J.data_ptr(), self.weights.data_ptr(), self.matrix.data_ptr(), self.sensor.data_ptr(),
            1, self.filter.closure.threshold, torch.get_num_threads(), self.statistics.data_ptr(),
            self.cell_sensor.data_ptr(), self.hits.data_ptr())
        self.seconds += time.perf_counter()-started
        self.calls += 1
        self.changed += int(self.statistics[0])
        self.maximum_sensor = max(self.maximum_sensor, float(self.statistics[1]))
        self.maximum_mean_error = max(self.maximum_mean_error, float(self.statistics[3]))
        if status:
            self.failed_state = state
            raise FloatingPointError('nonfinite state or invalid cell volume in modal filter')
        return state

    def summary(self):
        return dict(calls=self.calls, filtered_cell_calls=self.changed, seconds=self.seconds,
                    maximum_sensor=self.maximum_sensor, maximum_mean_error=self.maximum_mean_error)
