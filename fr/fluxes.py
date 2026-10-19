"""Face callback factories. Roe requires the equation's |A| jump action."""
from .core import central, rusanov_flux


def roe_flux(absolute_jacobian_action):
    return lambda m, p, axis: .5*(m.flux+p.flux)-.5*absolute_jacobian_action(p.state-m.state, m.state, p.state, axis)


def upwind_flux(velocity):
    """Scalar constant advection; systems should use their characteristic Roe action."""
    return lambda m, p, axis: m.flux if velocity[axis] >= 0 else p.flux


class LaxWendroffFlux:
    """Richtmyer interface predictor, composed with the chosen FR/RK method.

    This composition is not the classical finite-volume Lax-Wendroff scheme.
    prepare_step must receive the actual (possibly truncated) RK step.
    """
    def __init__(self, flux):
        self.flux = flux
        self.dt = None

    def prepare_step(self, dt, h):
        self.dt, self.h = dt, h

    def __call__(self, m, p, axis):
        if self.dt is None:
            raise RuntimeError('call prepare_step(dt, mesh.h) before evaluating LW flux')
        predicted = .5*(m.state+p.state)-.5*self.dt/self.h[axis]*(p.flux-m.flux)
        return self.flux(predicted, axis)


def make_numerical_flux(name, equation):
    """Select interface treatment separately from the physical equation.

    Custom callbacks pass through unchanged. Euler currently supplies Rusanov,
    central and Richtmyer choices; Roe requires an equation's |A| action.
    """
    if callable(name):
        return name
    choices = {'central': lambda: central,
               'rusanov': lambda: rusanov_flux(equation.speed),
               'lax-wendroff': lambda: LaxWendroffFlux(equation.flux)}
    if name in ('roe','upwind'):
        if not hasattr(equation,'absolute_jacobian_action'):
            raise ValueError('this equation does not implement characteristic Roe/upwind flux')
        return roe_flux(equation.absolute_jacobian_action)
    if name not in choices:
        raise ValueError(f'unknown numerical flux {name!r}')
    return choices[name]()
