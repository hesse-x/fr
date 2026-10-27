import unittest

import torch

from channel import Channel
from grouped_convection import GroupedConvection
from grouped_ldg import GroupedLDG
from mass_flux import ConstantMassFlux
from tensor_ops import ssprk43_add


class MassFluxTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Channel((4, 6, 3), forcing=0., re_tau=178.12,
                             lengths=(4*torch.pi, 4*torch.pi/3),
                             bulk_velocity=15.67873098359455,)

    def test_stage_constraint_and_energy_work(self):
        model = self.model
        state = model.initial(.5)
        raw = model.rhs(state)
        controlled = ConstantMassFlux(model, model.rhs)
        rhs = controlled(state)
        self.assertLess(abs(float(model.s.integral(rhs[..., 1:2])[0])), 1e-10)
        torch.testing.assert_close(rhs[..., 0], raw[..., 0])
        torch.testing.assert_close(rhs[..., 4]-raw[..., 4],
                                   controlled.last_force*state[..., 1]/state[..., 0])
        self.assertLess(abs(float(model.s.integral(state[..., 1:2])[0])/model.volume
                            -model.bulk_velocity), 1e-12)

    def test_native_multistep_constraint_and_perturbations(self):
        model = self.model
        state = model.initial(.5).movedim(-1, 0).contiguous().movedim(0, -1)
        native = GroupedLDG(model, (4, 4, 8)).make_rhs(
            GroupedConvection(model, (4, 2, 4)))
        controlled = ConstantMassFlux(model, native)
        reference = ConstantMassFlux(model, model.rhs)
        actual, expected = controlled(state), reference(state)
        torch.testing.assert_close(actual[..., :4], expected[..., :4], rtol=1e-10, atol=1e-8)
        # Independent Cartesian contractions and native FMA differ in the
        # cancellation of the large low-Mach total-energy flux.
        torch.testing.assert_close(actual[..., 4], expected[..., 4], rtol=1e-10, atol=1e-7)
        flux = model.s.integral(state[..., 1:2])[0]
        dt = model.timestep(state, .1)
        for step in range(12):
            state = ssprk43_add(controlled, state, step*dt, dt)
            if step in (2, 5, 8):
                state = model.perturb(state, .5, step)
            self.assertLess(abs(float(model.s.integral(state[..., 1:2])[0]-flux)), 1e-9)
            self.assertTrue(model.eq.valid(state))


if __name__ == '__main__':
    unittest.main()
