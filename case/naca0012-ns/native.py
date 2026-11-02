"""Production native NACA0012 adapter over geometry and load postprocessing."""
import importlib.util
from pathlib import Path
from fr.native.compressible import NativeCompressible

spec = importlib.util.spec_from_file_location('naca_native_ns_physics', Path(__file__).with_name('airfoil.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeAirfoil(NativeCompressible, module.Airfoil):
    pass
