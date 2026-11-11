"""CPU FP64 p=3 Cartesian NS preset; boundary sweeps are separate by default.

Imports never compile code. Plans compile lazily on construction. Kernels use
AVX2/FMA, Torch's CPU thread pool, private 4x4x4 group scratch and SoA fields.
"""
from .grouped_convection import GroupedConvection
from .grouped_ldg import GroupedLDG
from .grouped_fused import GroupedFused
from .native_step import NativeFluxStepper, NativeLocalRKStepper, NativeMeanAccumulator

__all__ = ['GroupedConvection', 'GroupedLDG', 'GroupedFused', 'NativeFluxStepper',
           'NativeLocalRKStepper', 'NativeMeanAccumulator']
