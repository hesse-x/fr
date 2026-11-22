"""Laminar, axisymmetric convection–diffusion field for a developed start."""
import numpy as np
import torch

from .generate_mesh import NOZZLE_RADIUS


def laminar_state(model):
    """Diffuse the nozzle profile downstream at speed Uj and recover radial entrainment."""
    geometry = model.mesh.volume_metrics().coordinates.numpy()
    x = geometry[..., 0].reshape(-1, len(model.edges)-1, 4, 4, 4)[0, :, 0, 0]
    yz = geometry[..., 1:].reshape(-1, len(model.edges)-1, 4, 4, 4, 2)
    radius = np.linalg.norm(yz, axis=-1)
    velocity = np.zeros(geometry.shape).reshape(*radius.shape, 3)
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    weights = np.outer(weights, weights)/np.pi
    radial_grid = np.linspace(0., min(radius.max(), NOZZLE_RADIUS+16*model.shear_width
                                      +8*np.sqrt(model.mu*model.edges[-1])), 1537)
    for layer, positions in enumerate(x):
        for k, downstream in enumerate(positions):
            # Gauss–Hermite quadrature of the transverse heat kernel.
            displacement = 2*np.sqrt(model.mu*downstream)*nodes
            along = radial_grid[:, None, None]+displacement[None, :, None]
            shifted = np.hypot(along, displacement[None, None, :])
            profile = np.tanh((shifted-NOZZLE_RADIUS)/model.shear_width)
            axial = ((1-profile)*.5*weights).sum(axis=(1, 2))
            direction = np.divide(along, shifted, out=np.zeros_like(shifted), where=shifted > 0)
            derivative = -(.5/model.shear_width*((1-profile**2)*direction*weights)).sum(axis=(1, 2))
            r = radius[:, layer, :, :, k]
            velocity[:, layer, :, :, k, 0] = np.interp(r, radial_grid, axial, right=0.)
            # Continuity gives v_r = -nu d(u_x)/dr for this diffusing profile.
            radial = np.interp(r, radial_grid, -model.mu*derivative, right=0.)
            factor = np.divide(radial, r, out=np.zeros_like(r), where=r > 0.)
            velocity[:, layer, :, :, k, 1:] = factor[..., None]*yz[:, layer, :, :, k]
    velocity = torch.from_numpy(velocity.reshape(geometry.shape)[model.stepper.order])
    density = torch.ones_like(velocity[..., 0])
    return model.eq.conservative(density, velocity, torch.full_like(density, model.pressure))
