import unittest

import torch

from channel import Channel
from grouped_fused import GroupedFused
from native_step import NativeFluxStepper, NativeLocalRKStepper, NativeMeanAccumulator


class RKBlockingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Channel((5, 6, 3), forcing=0., re_tau=178.12,
                             lengths=(4*torch.pi, 4*torch.pi/3), bulk_velocity=15.67873098359455,)
        self.plan = GroupedFused(self.model, (3, 4, 2))
        self.base = self.plan.make_rhs()
        self.reference = NativeFluxStepper(self.model, self.base)
        self.state = self.model.initial(.5).movedim(-1, 0).contiguous().movedim(0, -1)

    def test_local_multistep_and_pressure_constraint(self):
        local = NativeLocalRKStepper(self.model, self.plan)
        expected, actual = self.state.clone(), self.state.clone()
        initial_flux = float(self.model.s.integral(actual[..., 1:2])[0])
        original = self.state.clone()
        for step in range(12):
            dt = self.model.timestep(expected, .1)
            expected = self.reference.advance(expected, 0., dt)
            actual = local.advance(actual, 0., dt)
            if step in (2, 5, 8):
                expected = self.model.perturb(expected, .5, step)
                actual = self.model.perturb(actual, .5, step)

            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(local.last_force, self.reference.last_force)
            self.assertTrue(local.scan(actual, 1.2)[0])
            self.assertLess(abs(float(self.model.s.integral(actual[..., 1:2])[0])-initial_flux), 1e-9)
        self.assertTrue(torch.equal(self.state, original))

    def test_fused_finish_rejects_nonfinite_state(self):
        local = NativeLocalRKStepper(self.model, self.plan)
        dt = local.scan(self.state, .1)[1]
        invalid = self.state.clone()
        invalid[..., 4].reshape(-1)[0] = float('nan')
        _, valid, next_dt = local.advance_finish(invalid, 0., dt, .1)
        self.assertFalse(valid)
        self.assertTrue(torch.isnan(torch.tensor(next_dt)))

    def test_fused_finish(self):
        local = NativeLocalRKStepper(self.model, self.plan)
        mean = NativeMeanAccumulator(self.model)
        expected, actual = self.state.clone(), self.state.clone()
        modulation = 1+.01*torch.sin(torch.arange(4, dtype=torch.float64))
        expected.mul_(modulation.reshape(1, 1, 1, 4, 1, 1, 1))
        expected[..., 4].mul_(1.001)
        actual.copy_(expected)
        integral, fused_integral = mean.zeros(), mean.zeros()
        for step in range(8):
            dt = local.scan(expected, .1)[1]
            previous = expected
            expected = local.advance(previous, 0., dt)
            valid, next_dt = local.scan(expected, .1)
            sample = step >= 2
            if sample:
                mean.add(integral, previous, expected, dt)
            actual, fused_valid, fused_dt = local.advance_finish(
                actual, 0., dt, .1, fused_integral if sample else None)
            self.assertEqual(valid, fused_valid)
            self.assertEqual(next_dt, fused_dt)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(integral, fused_integral))
            if step == 3:
                expected = self.model.perturb(expected, .5, step)
                actual = self.model.perturb(actual, .5, step)


if __name__ == '__main__':
    unittest.main()
