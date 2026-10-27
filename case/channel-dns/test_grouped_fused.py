import unittest

import torch

from channel import Channel
from grouped_fused import GroupedFused
from native_step import NativeFluxStepper
from mass_flux import ConstantMassFlux
from tensor_ops import ssprk43_add


class GroupedFusedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_walls_and_ragged_groups(self):
        for cells in ((1, 1, 1), (5, 4, 3)):
            model = Channel(cells, stretch=1.7,)
            state = model.perturb(model.initial(), .4, 3)
            x, y, z = model.s.coords
            state *= (.9+.1*torch.cos(x)*torch.cos(2*z))[..., None]
            state[..., 4] += state[..., 0]*.02*model.temperature*(1-y*y)/(model.eq.gamma-1)
            state = state.movedim(-1, 0).contiguous().movedim(0, -1)
            expected = model.rhs(state)
            plan = GroupedFused(model, (2, 3, 2))
            torch.testing.assert_close(plan(state), expected, rtol=1e-10, atol=1e-7)

    def test_fused_multistep_constant_flux(self):
        model = Channel((4, 6, 3), forcing=0., re_tau=178.12,
                        lengths=(4*torch.pi, 4*torch.pi/3), bulk_velocity=15.67873098359455,)
        state = model.initial(.5).movedim(-1, 0).contiguous().movedim(0, -1)
        plan = GroupedFused(model, (3, 4, 2))
        native = NativeFluxStepper(model, plan.make_rhs())
        reference_rhs = ConstantMassFlux(model, model.rhs)
        reference, actual = state, state
        for step in range(12):
            dt = model.timestep(reference, .1)
            reference = ssprk43_add(reference_rhs, reference, 0., dt)
            actual = native.advance(actual, 0., dt)
            if step in (2, 5, 8):
                reference = model.perturb(reference, .5, step)
                actual = model.perturb(actual, .5, step)
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-10)
            self.assertTrue(native.scan(actual, 1.2)[0])
        flux = float(model.s.integral(actual[..., 1:2])[0])/model.volume
        self.assertAlmostEqual(flux, model.bulk_velocity, places=11)


if __name__ == '__main__':
    unittest.main()
