"""Cylinder adapter for the shared native compressible-flow operators."""
from fr.native.compressible import NativeCompressible, load_operators
from cylinder import Cylinder


class NativeCylinder(NativeCompressible, Cylinder):
    implementation = 'torch.ops.cylinder_flow CPU FP64 C++'
