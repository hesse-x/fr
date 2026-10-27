"""Diameter-one cylinder using the NACA0012 laminar NS discretization."""
import importlib.util
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    'cylinder_ns_physics', ROOT/'case/naca0012-ns/airfoil.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Cylinder(module.Airfoil):
    # The shared solver normalizes viscosity and forces by a unit length.
    # Here that length is the cylinder diameter D=1, rather than the chord.
    def __init__(self, mesh, p=2, mach=.1, alpha=0., quadrature_order=6,
                 reynolds=100., **kwargs):
        super().__init__(mesh, p, mach, alpha, quadrature_order,
                         reynolds=reynolds, **kwargs)

    def initial_state(self):
        U = self.far.expand(self.s.ndof,4).clone()
        x, y = self.s.coords
        # A localized cross-stream seed breaks reflection symmetry downstream.
        dv = .001*self.mach*torch.exp(-((x-1.5)**2+y**2)/.25)
        U[:,3] += U[:,2]*dv+.5*U[:,0]*dv**2
        U[:,2] += U[:,0]*dv
        return U
