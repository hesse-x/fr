import unittest
import torch
from channel import Channel
from grouped_convection import GroupedConvection
from tensor_ops import ssprk43_add


class GroupedConvectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_interfaces_and_ragged_groups(self):
        for cells in ((1, 1, 1), (2, 3, 2), (5, 4, 3)):
            with self.subTest(cells=cells):
                model = Channel(cells, stretch=1.7,)
                state = model.perturb(model.initial(), .4, 3)
                generator = torch.Generator().manual_seed(42)
                density = .8+.4*torch.rand(state.shape[:-1]+(1,), generator=generator, dtype=state.dtype)
                state = (state*density).movedim(-1, 0).contiguous().movedim(0, -1)
                grouped = GroupedConvection(model, (2, 2, 2))
                actual = grouped(state)
                reference = model.convection(state, fields=model.eq.primitive(state))
                torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-10)
                self.assertEqual(actual.stride(), state.stride())
                # Reflecting walls and periodic interfaces conserve mass.
                self.assertLess(abs(float(model.s.integral(actual)[0])), 1e-10)

    def test_complete_rhs_and_evolution(self):
        model = Channel((4, 5, 3), stretch=1.7,)
        state = model.perturb(model.initial(), .4, 3).movedim(-1, 0).contiguous().movedim(0, -1)
        expected_rhs = model.rhs(state)
        dt = model.timestep(state)
        expected = state
        for i in range(4):
            expected = ssprk43_add(model.rhs, expected, i*dt, dt)
        model.convection = GroupedConvection(model, (2, 3, 2))
        actual_rhs = model.rhs(state)
        torch.testing.assert_close(actual_rhs[..., :4], expected_rhs[..., :4], atol=1e-8, rtol=1e-10)
        # FMA changes cancellation in the low-Mach total-energy flux (~1e7).
        # Keep the tighter multistep state tolerance below unchanged.
        torch.testing.assert_close(actual_rhs[..., 4], expected_rhs[..., 4], atol=1e-7, rtol=1e-10)
        actual = state
        for i in range(4):
            actual = ssprk43_add(model.rhs, actual, i*dt, dt)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)
        self.assertTrue(model.eq.valid(actual))


if __name__ == '__main__':
    unittest.main()
