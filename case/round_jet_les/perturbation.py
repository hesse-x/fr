"""One-time transverse shear-layer seeds for a mapped round jet."""
import math

import torch

from .generate_mesh import NOZZLE_RADIUS


class ShearSeeds:
    def __init__(self, model, steps, amplitude, start_step, saved):
        self.model = model
        self.steps = tuple(sorted(steps))
        self.amplitude = amplitude
        self.applied = []
        if saved is not None:
            if self.steps and (self.steps != tuple(saved['steps']) or amplitude != saved['amplitude']):
                raise ValueError('shear seed schedule differs from the checkpoint')
            self.steps = tuple(saved['steps'])
            self.amplitude = saved['amplitude']
            self.applied = saved['applied'].copy()
        if (any(step < 0 for step in self.steps) or len(set(self.steps)) != len(self.steps)
                or not math.isfinite(self.amplitude) or self.amplitude <= 0.):
            raise ValueError('shear seeds require unique nonnegative steps and positive amplitude')
        required = {step for step in self.steps if step < start_step}
        if (not required.issubset(self.applied)
                or any(step not in self.steps or step > start_step for step in self.applied)):
            raise ValueError('shear seed history is inconsistent with the starting step')

    def checkpoint(self):
        return dict(steps=list(self.steps), amplitude=self.amplitude, applied=self.applied)

    def apply(self, state, step):
        if step not in self.steps or step in self.applied:
            return False
        phase_rotation = .7*self.steps.index(step)
        order = self.model.stepper.order.tolist()
        x, y, z = self.model.mesh.volume_metrics(cells=order).coordinates.unbind(-1)
        radius, theta = torch.hypot(y, z), torch.atan2(z, y)
        width = self.model.shear_width
        window = torch.exp(-.5*((radius-NOZZLE_RADIUS)/width).square())
        envelope = torch.where((x>3.) & (x<9.), torch.sin(torch.pi*(x-3.)/6.).square(), 0.)
        radial, tangent = torch.zeros_like(x), torch.zeros_like(x)
        # The two velocity components are transverse derivatives of a streamfunction.
        for mode, strength, axial_wave, phase in ((3,1.,4.,.2),(4,.65,-7.,1.1),(5,.4,9.,2.3)):
            angle = mode*(theta-phase_rotation)+axial_wave*x+phase
            profile = strength*envelope*window*(radius/NOZZLE_RADIUS)**(mode-1)/NOZZLE_RADIUS
            radial += profile*mode*angle.cos()
            tangent -= profile*(mode-radius*(radius-NOZZLE_RADIUS)/width**2)*angle.sin()
        peak = float(torch.hypot(radial,tangent).max())
        if not math.isfinite(peak) or peak <= 0.:
            raise ValueError('no solution points available in the shear seed region')
        scale = self.amplitude/peak
        dy = scale*(radial*theta.cos()-tangent*theta.sin())
        dz = scale*(radial*theta.sin()+tangent*theta.cos())
        rho, _, my, mz, energy = state.unbind(-1)
        energy.add_(my*dy+mz*dz+.5*rho*(dy.square()+dz.square()))
        my.add_(rho*dy)
        mz.add_(rho*dz)
        self.applied.append(step)
        return True
