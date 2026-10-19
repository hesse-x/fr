"""Configurable tensor-product FR solvers."""
from .core import CartesianFR, Face, central, rusanov_flux
from .equations import Euler, LinearizedEuler, Equation, make_equation
from .maxwell import Maxwell
from .time import integrate, rk4, ssprk43

__all__=['CartesianFR','Face','central','rusanov_flux','Euler','Maxwell','integrate','rk4','ssprk43']

from .acoustics import Acoustics, ConvectedAcoustics
from .pml import SplitPML
from .fluxes import roe_flux, upwind_flux, LaxWendroffFlux, make_numerical_flux
from .boundaries import extrapolate, prescribed, per_face, acoustic_wall

__all__ += ['Equation','LinearizedEuler','make_equation','make_numerical_flux','Acoustics','ConvectedAcoustics','SplitPML','roe_flux','upwind_flux','LaxWendroffFlux',
            'extrapolate','prescribed','per_face','acoustic_wall']

from .mesh import Mesh2D
from .unstructured import MeshFR2D, NormalFace, normal_central, normal_rusanov_flux, make_normal_numerical_flux
__all__ += ['Mesh2D','MeshFR2D','NormalFace','normal_central','normal_rusanov_flux','make_normal_numerical_flux']

from .second_order import CartesianLDG
__all__ += ['CartesianLDG']

from .equations.schrodinger import Schrodinger
__all__ += ['Schrodinger']
