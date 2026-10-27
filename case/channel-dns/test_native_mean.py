import unittest

import torch

from channel import Channel
from native_step import NativeMeanAccumulator


class NativeMeanTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Channel((3, 5, 2), stretch=1.7)
        self.native = NativeMeanAccumulator(self.model)

    def test_variable_density_temperature_and_multiple_intervals(self):
        model = self.model
        x, y, z = model.s.coords
        density = .9+.1*torch.cos(x)*torch.cos(2*z)
        previous = (model.initial(.5)*density[..., None]).movedim(-1, 0).contiguous().movedim(0, -1)
        integral = self.native.zeros()
        expected = torch.zeros_like(integral)
        for event, dt in enumerate((.001, .0007, .0014), 1):
            current = model.perturb(previous, .3, event)
            current[..., 4] += current[..., 0]*.03*model.temperature*(1-y*y)/(model.eq.gamma-1)
            expected.add_(model.field_moments(previous), alpha=.5*dt)
            expected.add_(model.field_moments(current), alpha=.5*dt)
            self.native.add(integral, previous, current, dt)
            torch.testing.assert_close(integral, expected, rtol=1e-13, atol=1e-12)
            previous = current

    def test_reject_incompatible_layout_and_duration(self):
        state = self.model.initial().movedim(-1, 0).contiguous().movedim(0, -1)
        integral = self.native.zeros()
        for dt in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.native.add(integral, state, state, dt)
        with self.assertRaises(ValueError):
            self.native.add(integral.contiguous(), state, state, .001)


if __name__ == '__main__':
    unittest.main()
