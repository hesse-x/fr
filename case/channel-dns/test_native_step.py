import unittest

import torch

from channel import Channel
from grouped_convection import GroupedConvection
from grouped_ldg import GroupedLDG
from mass_flux import ConstantMassFlux
from native_step import NativeFluxStepper
from tensor_ops import ssprk43_add


class NativeStepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Channel((4, 6, 3), re_tau=178.12, forcing=0.,
                             lengths=(4*torch.pi, 4*torch.pi/3), bulk_velocity=15.67873098359455,)
        self.base = GroupedLDG(self.model, (4, 4, 8)).make_rhs(
            GroupedConvection(self.model, (4, 2, 4)))
        self.native = NativeFluxStepper(self.model, self.base)
        self.state = self.model.initial(.5).movedim(-1, 0).contiguous().movedim(0, -1)

    def test_scan_and_invalid_states(self):
        good, dt = self.native.scan(self.state, 1.2)
        self.assertTrue(good)
        self.assertAlmostEqual(dt/self.model.timestep(self.state, 1.2), 1., places=13)
        for variable, value in ((0, -1.), (0, 0.), (4, -1.), (1, float('nan')),
                                (2, float('inf')), (3, float('-inf'))):
            with self.subTest(variable=variable, value=value):
                state = self.state.clone()
                state[0, 0, 0, 0, 0, 0, variable] = value
                self.assertFalse(self.model.eq.valid(state))
                self.assertFalse(self.native.scan(state, 1.2)[0])

    def test_multistep_flux_and_energy(self):
        reference_rhs = ConstantMassFlux(self.model, self.base)
        reference, actual = self.state.clone(), self.state.clone()
        flux = self.model.s.integral(actual[..., 1:2])[0]
        for step in range(16):
            dt = self.model.timestep(reference, .1)
            reference = ssprk43_add(reference_rhs, reference, step*dt, dt)
            actual = self.native.advance(actual, step*dt, dt)
            if step in (3, 7, 11):
                reference = self.model.perturb(reference, .5, step)
                actual = self.model.perturb(actual, .5, step)
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-10)
            self.assertLess(abs(float(self.model.s.integral(actual[..., 1:2])[0]-flux)), 1e-9)
            self.assertTrue(self.native.scan(actual, 1.2)[0])


if __name__ == '__main__':
    unittest.main()
