"""Shared physical equations; cases choose models independently of FR and RK."""
from typing import Protocol
import torch
from .schrodinger import Schrodinger
from .euler import Euler, GAMMA, primitive, flux, rusanov
from .linearized_euler import LinearizedEuler
from .maxwell import Maxwell, MaxwellTMz
from .acoustics import Acoustics, ConvectedAcoustics


class Equation(Protocol):
    nvars: int

    def flux(self, U: torch.Tensor, axis: int) -> torch.Tensor: ...
    def speed(self, U: torch.Tensor, axis: int) -> torch.Tensor: ...
    def valid(self, U: torch.Tensor) -> bool: ...


def make_equation(name, **parameters):
    """Construct a shared model. State layouts remain equation-specific."""
    models = {'euler': Euler, 'linearized-euler': LinearizedEuler,
              'maxwell': Maxwell, 'maxwell-tmz': MaxwellTMz, 'acoustics': Acoustics,
              'convected-acoustics': ConvectedAcoustics, 'schrodinger': Schrodinger}
    key = name.lower().replace('_','-')
    if key not in models:
        raise ValueError(f'unknown equation {name!r}; choose {", ".join(models)}')
    return models[key](**parameters)


__all__ = ['Schrodinger','Equation','Euler','LinearizedEuler','Maxwell','MaxwellTMz','Acoustics',
           'ConvectedAcoustics','make_equation','GAMMA','primitive','flux','rusanov']
