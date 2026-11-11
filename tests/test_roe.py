"""Roe wave resolution, conservative FR coupling and native RK parity."""
import tempfile
from pathlib import Path
import unittest
import torch
from fr.core import Face
from fr.fluxes import make_numerical_flux
from fr.equations.navier_stokes import NavierStokes
from case.taylor_green_vertox_les import TaylorGreenVortexLES, run


class RoeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_consistency_and_stationary_contact_shear(self):
        eq = NavierStokes()
        roe, rus = (make_numerical_flux(name, eq) for name in ('roe', 'rusanov'))
        for axis in range(3):
            velocity = torch.tensor([.2, .3, -.1], dtype=torch.float64)
            velocity[axis] = 0.
            right_velocity = 2*velocity
            left = eq.conservative(1., velocity, 1.)
            right = eq.conservative(2., right_velocity, 1.)
            fl, fr = eq.flux(left, axis), eq.flux(right, axis)
            torch.testing.assert_close(roe(Face(left, fl), Face(left, fl), axis), fl)
            torch.testing.assert_close(roe(Face(left, fl), Face(right, fr), axis), fl, atol=2e-15, rtol=0.)
            self.assertGreater(float((rus(Face(left, fl), Face(right, fr), axis)-fl).abs().max()), .1)

    def test_acoustic_entropy_fix(self):
        eq = NavierStokes()
        sound = (eq.gamma)**.5
        velocity = torch.tensor([sound, .2, -.3], dtype=torch.float64)
        state = eq.conservative(1., velocity, 1.)
        h = (state[-1]+1.)/state[0]
        wave = torch.cat((state.new_tensor([1.]), velocity-state.new_tensor([sound,0.,0.]), (h-sound*velocity[0]).reshape(1)))
        actual = eq.absolute_jacobian_action(wave, state, state, 0)
        torch.testing.assert_close(actual, .025*sound*wave, rtol=1e-13, atol=1e-14)

    def test_native_torch_residual_and_multistep(self):
        for mode in ('separate', 'fused'):
            model = TaylorGreenVortexLES((5, 3, 4), interface_flux='roe')
            state = model.initial()
            native = model.make_stepper('native-p3', block=(2, 2, 3), boundary_mode=mode)
            torch_step = model.make_stepper('torch')
            torch.testing.assert_close(native.plan(state), model.rhs(state), rtol=1e-10, atol=3e-12)
            torch.testing.assert_close(native.plan.convection(state), model.convection(state), rtol=1e-10, atol=3e-12)
            torch.testing.assert_close(model.s.integral(native.plan(state)), state.new_zeros(5), rtol=0, atol=2e-12)
            actual = expected = state
            dt = model.timestep(state, .2)
            for i in range(5):
                actual, valid, _ = native.advance_finish(actual, i*dt, dt, .2)
                expected = torch_step(expected, i*dt, dt)
                self.assertTrue(valid)
                torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-13)
            torch.testing.assert_close(model.s.integral(actual), model.s.integral(state), atol=3e-11, rtol=1e-13)
            self.assertEqual(native.plan.metadata['interface_flux'], 'roe')

    def test_restart_rejects_interface_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'roe'
            options = dict(cells=(2,)*3, p=2, backend='torch', interface_flux='roe',
                           final_time=.002, max_dt=.001, snapshot_times=(), plot=False, fields=False)
            run(output, max_steps=1, **options)
            actual, _ = run(output, resume=True, **options)
            reference, _ = run(Path(temporary)/'direct', **options)
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            options['interface_flux'] = 'rusanov'
            with self.assertRaises(ValueError):
                run(output, resume=True, **options)
