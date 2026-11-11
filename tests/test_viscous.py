"""Viscous schemes: consistency, energy, conservation and fused native parity."""
import math
import unittest
import torch

from fr import (CartesianFR, CartesianBR1Penalty, CartesianNavierStokes,
                Mesh2D, MeshFR2D, NavierStokes, ViscousConfig,
                central, normal_central, make_viscous_operator, ssprk43)
from fr.native.cartesian_ns import GroupedFused, GroupedLDG


class ViscousTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_invalid_configuration(self):
        for kw in ({'scheme': 'br2'}, {'penalty': 0.}, {'penalty': float('nan')}):
            with self.assertRaises(ValueError):
                ViscousConfig(**kw)

    def test_cartesian_br1_energy_and_polynomial(self):
        torch.manual_seed(123)
        for dim in (1, 2, 3):
            for periodic in (True, False):
                s = CartesianFR((2,)*dim, ((0., 1.),)*dim, 3,
                    flux=lambda U,a: U, numerical_flux=central, periodic=periodic,
                    boundary=lambda c,U,a,side,t: U,
                    edges=([0., .3, 1.],)*dim)
                op = CartesianBR1Penalty(s, lambda c,U,a,side,t: torch.zeros_like(U))
                U = torch.randn(*s.shape, 2, dtype=torch.float64)
                diffusion = op.diffusion(U)
                self.assertLess(float(s.integral(U*diffusion).max()), 0.)
                # Central derivative pair is also negative-adjoint with zero data.
                F = torch.randn(*s.shape, 2, dim, dtype=U.dtype)
                torch.testing.assert_close(s.integral(U*op.divergence(F)),
                    -s.integral((op.gradient(U)*F).sum(-1)), atol=1e-11, rtol=1e-12)
                if periodic:
                    torch.testing.assert_close(s.integral(diffusion), torch.zeros(2, dtype=U.dtype),
                                               atol=2e-10, rtol=0.)
                else:
                    def value(coords):
                        return sum(x*x for x in coords)[..., None]
                    op = CartesianBR1Penalty(s, lambda c,U,a,side,t: value(c))
                    torch.testing.assert_close(op.diffusion(value(s.coords)),
                        torch.full_like(value(s.coords), 2*dim), atol=2e-10, rtol=0.)

    def test_cartesian_br1_fourier_convergence(self):
        errors = []
        for n in (4, 8, 16):
            s = CartesianFR((n,), ((0., 2*math.pi),), 3,
                flux=lambda U,a: U, numerical_flux=central, periodic=True)
            op = CartesianBR1Penalty(s)
            U = torch.sin(s.coords[0])[..., None]
            errors.append(float(s.norm(op.diffusion(U)+U)[0]))
        self.assertGreater(math.log2(errors[-2]/errors[-1]), 1.8)

    def test_cartesian_br1_native_reference_and_rk(self):
        for cells, wall, penalty in (((1, 1, 1), False, 4.),
                                     ((3, 4, 2), False, 2.),
                                     ((3, 4, 2), True, 4.)):
            edges = (None, [0., .1, .4, .7, 1.], None) if cells[1] == 4 else None
            a = CartesianNavierStokes(cells, ((0., 1.),)*3, viscous_scheme='br1-penalty',
                penalty=penalty, mu=.03, boundary='isothermal-channel' if wall else 'periodic',
                wall_temperature=2. if wall else None, edges=edges)
            x, y, z = a.s.coords
            velocity = torch.stack((.1*torch.sin(2*math.pi*y), .1*torch.cos(2*math.pi*z),
                                    .1*torch.sin(2*math.pi*x)), -1)
            U = a.soa(a.eq.conservative(1+.02*torch.sin(2*math.pi*x), velocity,
                                        2.+.01*torch.cos(2*math.pi*y)))
            expected = a.rhs(U)
            if not wall:
                torch.testing.assert_close(a.s.integral(expected), torch.zeros(5, dtype=U.dtype),
                                           atol=2e-10, rtol=0.)
            for mode in ('separate', 'fused'):
                plan = GroupedFused(a, (2, 3, 2), boundary_mode=mode)
                torch.testing.assert_close(plan(U), expected, atol=3e-10, rtol=2e-11)
                independent = GroupedLDG(a, (2, 3, 2), boundary_mode=mode)
                torch.testing.assert_close(independent(U, a.convection(U)), expected,
                                           atol=3e-10, rtol=2e-11)
                step = a.make_stepper('native-p3', block=(2, 3, 2), boundary_mode=mode,
                                     constant_mass_flux=wall)
                reference = a.make_stepper('torch', constant_mass_flux=wall)
                actual, exact = U, U
                for i in range(3):
                    dt = a.timestep(actual, .03)
                    actual, valid, next_dt = step.advance_finish(actual, i*dt, dt, .03)
                    exact = reference(exact, i*dt, dt)
                    self.assertTrue(valid)
                    torch.testing.assert_close(actual, exact, atol=3e-12, rtol=2e-13)
                    self.assertAlmostEqual(next_dt/a.timestep(actual, .03), 1., places=12)

    def mesh_grid(self, kind):
        mesh = Mesh2D.rectangle(2, kind=kind, jitter=.1).mapped(
            lambda x,y: (x+.03*torch.sin(math.pi*x)*torch.sin(math.pi*y),
                         y+.02*torch.sin(math.pi*x)*torch.sin(math.pi*y)), 3)
        eq = NavierStokes(2, mu=.1)
        s = MeshFR2D(mesh, p=3, flux=eq.flux, numerical_flux=normal_central,
                     boundary=lambda c,U,n,tag,t: U, quadrature_order=8)
        return s, eq

    def test_curved_mixed_affine_gradient_and_ns_manufactured_rhs(self):
        for kind in ('tri', 'quad', 'mixed'):
            s, eq = self.mesh_grid(kind)
            def primitive(coords):
                x, y = coords
                return torch.stack((.2+x+2*y, -.1+3*x-y, 2.+.3*x+.4*y), -1)
            W = primitive(s.coords)
            U = eq.conservative(1., W[..., :2], W[..., 2])
            gradient = U.new_tensor([[1., 2.], [3., -1.], [.3, .4]]).expand(s.ndof, -1, -1)
            for scheme in ('ldg', 'br1-penalty'):
                op = make_viscous_operator(s, eq, scheme=scheme,
                    boundary=lambda c,W,n,tag,t: primitive(c))
                torch.testing.assert_close(op.gradient(W), gradient, atol=3e-9, rtol=1e-10)
                residual, faces = op.evaluate(U)
                # Constant stress: momentum divergence=0, energy divergence=tau:grad(u).
                stress = eq.viscous_flux_from_primitive(W, gradient)[..., 1:3, :]
                exact = torch.zeros_like(U)
                exact[..., -1] = (stress*gradient[..., :2, :]).sum((-1, -2))
                torch.testing.assert_close(residual, exact, atol=3e-8, rtol=1e-9)
                boundary = s.mesh.boundary
                total = (faces[boundary]*s.face_jacobian[boundary, :, None]*s.fw[None, :, None]).sum((0, 1))
                torch.testing.assert_close(s.integral(residual), total, atol=1e-9, rtol=1e-10)

    def test_curved_jump_conservation_and_kinetic_dissipation(self):
        torch.manual_seed(321)
        for kind in ('tri', 'quad', 'mixed'):
            s, eq = self.mesh_grid(kind)
            velocity = .1*torch.randn(s.ndof, 2, dtype=torch.float64)
            U = eq.conservative(1., velocity, torch.ones(s.ndof, dtype=velocity.dtype))
            def boundary(c,W,n,tag,t):
                value = torch.zeros_like(W)
                value[..., -1] = 1.
                return value
            for scheme in ('ldg', 'br1-penalty'):
                op = make_viscous_operator(s, eq, scheme=scheme, boundary=boundary)
                residual, flux = op.evaluate(U)
                self.assertLess(float(s.integral((velocity*residual[..., 1:3]).sum(-1, keepdim=True))[0]), 0.)
                interior = ~s.mesh.boundary
                torch.testing.assert_close(flux[interior], -s.neighbor_states(flux)[interior],
                                           atol=3e-11, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
