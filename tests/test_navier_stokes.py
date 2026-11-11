"""Shared constitutive laws and Cartesian NS execution presets."""
import math
import unittest

import torch
from fr import CartesianNavierStokes, NavierStokes, make_equation, ssprk43
from fr.native.cartesian_ns import GroupedFused, NativeMeanAccumulator


class NavierStokesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_constitutive_stress_heat_and_state(self):
        eq = make_equation('navier-stokes', dim=3, gamma=1.4, mu=.2, prandtl=.71)
        self.assertIsInstance(eq, NavierStokes)
        velocity = torch.tensor([[2., -1., 3.]], dtype=torch.float64)
        U = eq.conservative(2., velocity, 10.)
        rho, recovered, pressure = eq.primitive(U)
        torch.testing.assert_close(rho, torch.tensor([2.], dtype=torch.float64))
        torch.testing.assert_close(pressure, torch.tensor([10.], dtype=torch.float64))
        torch.testing.assert_close(recovered, velocity)
        gradient = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],
                                  [.5, -.3, .2]]], dtype=torch.float64)
        flux = eq.viscous_flux(U, gradient)
        expected = .2*torch.tensor([[[-8., 6., 10.], [6., 0., 14.], [10., 14., 8.]]], dtype=torch.float64)
        torch.testing.assert_close(flux[..., 1:4, :], expected)
        torch.testing.assert_close(flux[..., -1, :], (velocity[..., :, None]*expected).sum(-2)+eq.kappa*gradient[..., -1, :])
        self.assertEqual(float(flux[..., 0, :].abs().sum()), 0.)
        # Stokes hypothesis: isotropic expansion has zero deviatoric stress.
        gradient.zero_()
        gradient[..., :3, :] = torch.eye(3, dtype=gradient.dtype)*2
        torch.testing.assert_close(eq.viscous_flux(U, gradient), torch.zeros_like(flux), atol=1e-15, rtol=0.)

    def periodic_model(self, cells=(5, 4, 3), **kwargs):
        model = CartesianNavierStokes(cells, ((-math.pi, math.pi),)*3,
                                     mu=.01, gamma=1.4, **kwargs)
        x, y, z = model.s.coords
        velocity = torch.stack((.7+.1*torch.sin(y), -.2+.1*torch.cos(z), .3+.1*torch.sin(x)), -1)
        rho = 1+.04*torch.cos(x)*torch.sin(z)
        pressure = 25+.1*torch.cos(y)
        return model, model.soa(model.eq.conservative(rho, velocity, pressure))

    def test_periodic_rhs_conservation_and_native_reference(self):
        for cells in ((1, 1, 1), (5, 4, 3)):
            model, state = self.periodic_model(cells)
            expected = model.rhs(state)
            torch.testing.assert_close(model.s.integral(expected), torch.zeros(5, dtype=state.dtype), atol=2e-11, rtol=0.)
            for mode in ('fused', 'separate'):
                plan = GroupedFused(model, (3, 2, 2), boundary_mode=mode)
                torch.testing.assert_close(plan(state), expected, rtol=2e-11, atol=2e-11)
                # Periodic seams must be neighbors, never false wall metadata.
                self.assertEqual(int(plan.ldg.buffers[2][..., 3:].sum()), 0)

    def test_periodic_multistep_without_mass_flux_control(self):
        model, state = self.periodic_model()
        native = model.make_stepper('native-p3', block=(3, 2, 2))
        self.assertFalse(native.constant_mass_flux)
        self.assertEqual(native.plan.metadata['boundary_mode'], 'separate')
        reference = state
        actual = state
        means = NativeMeanAccumulator(model).zeros()
        expected_means = NativeMeanAccumulator(model).zeros()
        dt = model.timestep(state, .1)
        total = model.s.integral(state)
        for step in range(6):
            previous = actual
            reference = ssprk43(model.rhs, reference, step*dt, dt)
            actual, valid, next_dt = native.advance_finish(actual, step*dt, dt, .1, means)
            self.assertTrue(valid)
            NativeMeanAccumulator(model).add(expected_means, previous, actual, dt)
            torch.testing.assert_close(actual, reference, rtol=2e-13, atol=1e-12)
            self.assertAlmostEqual(next_dt/model.timestep(actual, .1), 1., places=13)
        torch.testing.assert_close(means, expected_means, rtol=0., atol=0.)
        torch.testing.assert_close(model.s.integral(actual), total, rtol=2e-14, atol=3e-11)
        self.assertEqual(native.last_force, 0.)
        torch.testing.assert_close(native(state, 0., dt), native.advance(state, 0., dt), rtol=0., atol=0.)

    def test_prescribed_force_and_torch_preset(self):
        model, state = self.periodic_model((2, 3, 2), forcing=.3)
        native = model.make_stepper('native-p3')
        reference_step = model.make_stepper('torch')
        dt = model.timestep(state, .1)
        actual = native(state, 0., dt)
        expected = reference_step(state, 0., dt)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=1e-12)
        momentum_change = model.s.integral(actual-state)[1]
        self.assertAlmostEqual(float(momentum_change)/(dt*model.volume), .3, places=12)
        with self.assertRaises(ValueError):
            model.make_stepper('native-p3', constant_mass_flux=True)

    def test_wall_preset_and_constant_flux(self):
        model = CartesianNavierStokes((3, 5, 2), ((0., 2.), (-1., 1.), (0., 3.)),
                                     boundary='isothermal-channel', wall_temperature=30., mu=.01)
        x, y, z = model.s.coords
        velocity = torch.stack((1-y*y, torch.zeros_like(x), torch.zeros_like(z)), -1)
        state = model.soa(model.eq.conservative(1., velocity, 30.))
        native = model.make_stepper('native-p3', constant_mass_flux=True, block=(2, 3, 2))
        torch.testing.assert_close(native.plan(state), model.rhs(state), rtol=1e-11, atol=1e-11)
        expected = model.make_stepper('torch', constant_mass_flux=True)
        dt = model.timestep(state, .1)
        torch.testing.assert_close(native(state, 0., dt), expected(state, 0., dt), rtol=1e-13, atol=1e-12)

    def test_reject_unsupported_native_and_invalid_parameters(self):
        for parameters in ({'mu': 0.}, {'mu': float('nan')}, {'prandtl': 0.}, {'gamma': float('nan')}):
            with self.assertRaises(ValueError):
                NavierStokes(**parameters)
        for parameters in ({'p': 2},):
            model, state = self.periodic_model((2, 2, 2), **parameters)
            with self.assertRaises(ValueError):
                model.make_stepper('native-p3')
            self.assertTrue(model.eq.valid(model.make_stepper('torch')(state, 0., 1e-5)))
        model, state = self.periodic_model((2, 2, 2))
        with self.assertRaises(ValueError):
            model.make_stepper('unknown')
        with self.assertRaises(ValueError):
            model.make_stepper('native-p3', boundary_mode='unknown')
        with self.assertRaises(ValueError):
            model.make_stepper('native-p3', block=(0, 4, 4))
        native = model.make_stepper('native-p3')
        with self.assertRaises(ValueError):
            native(state.contiguous(), 0., .001)
        with self.assertRaises(ValueError):
            native(state, 0., -.001)
        with self.assertRaises(ValueError):
            native.scan(state, 0.)
        with self.assertRaises(ValueError):
            CartesianNavierStokes((2, 2, 2), ((0., 1.),)*3, boundary='isothermal-channel')


if __name__ == '__main__':
    unittest.main()
