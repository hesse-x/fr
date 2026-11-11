"""Paired fused/separate-boundary regression, including local RK and means."""
import unittest

import torch

from channel import Channel
from grouped_fused import GroupedFused
from native_step import NativeLocalRKStepper, NativeMeanAccumulator


class BoundarySplitTests(unittest.TestCase):
    def test_rhs_local_rk_and_means(self):
        torch.set_num_threads(2)
        for cells in ((1, 1, 1), (5, 6, 3)):
            model = Channel(cells, forcing=0., stretch=1.7)
            state = model.perturb(model.initial(), .4, 3)
            x, y, z = model.s.coords
            state *= (.9+.1*torch.cos(x)*torch.cos(2*z))[..., None]
            state[..., -1] += state[..., 0]*.02*model.temperature*(1-y*y)/(model.eq.gamma-1)
            state = state.movedim(-1, 0).contiguous().movedim(0, -1)
            plans = []
            for mode in ('fused', 'separate'):
                plans.append(GroupedFused(model, (3, 4, 2), boundary_mode=mode))
            reference = model.rhs(state)
            for plan in plans:
                torch.testing.assert_close(plan(state), reference, rtol=1e-10, atol=1e-7)
            torch.testing.assert_close(plans[0](state), plans[1](state), rtol=0., atol=0.)
            steppers = [NativeLocalRKStepper(model, plan) for plan in plans]
            means = [NativeMeanAccumulator(model).zeros() for _ in plans]
            states = [state.clone(memory_format=torch.preserve_format) for _ in plans]
            valid, dt = steppers[0].scan(state, .2)
            self.assertTrue(valid)
            for step in range(8):
                results = [native.advance_finish(u, step*dt, dt, .2, mean)
                           for native, u, mean in zip(steppers, states, means)]
                self.assertTrue(all(result[1] for result in results))
                states = [result[0] for result in results]
                torch.testing.assert_close(states[0], states[1], rtol=0., atol=0.)
                torch.testing.assert_close(means[0], means[1], rtol=0., atol=0.)
                self.assertEqual(results[0][2], results[1][2])


if __name__ == '__main__':
    unittest.main()
