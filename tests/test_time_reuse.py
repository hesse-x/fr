import unittest

import torch

from fr.time import ssprk43, ssprk43_stages


class ReusedRKTests(unittest.TestCase):
    def test_cached_first_rhs_with_local_steps_and_stage_times(self):
        state = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        dt = torch.tensor([[.01], [.03]], dtype=torch.float64)
        calls = []

        def rhs(u, t):
            calls.append(t)
            return -u+t

        expected = ssprk43(rhs, state, .7, dt)
        first = rhs(state, .7)
        calls.clear()
        actual = ssprk43(rhs, state, .7, dt, first_rhs=first)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), 3)
        for actual_time, stage in zip(calls, ssprk43_stages(.7, dt)[1:]):
            torch.testing.assert_close(actual_time, stage[0])
        torch.testing.assert_close(state, torch.tensor([[1., 2.], [3., 4.]], dtype=state.dtype))
