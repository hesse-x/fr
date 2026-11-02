import unittest

import torch

from case.parabolic_antenna.solve import ParabolicAntennaCase
from case.parabolic_antenna.native import NativeMaxwell
from fr.time import rk4


class NativeParabolicTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)

    def test_linear_field_curl_generic_orders(self):
        for p in (2, 3, 4):
            with self.subTest(p=p):
                case = ParabolicAntennaCase(refinement=1, p=p)
                physical = torch.zeros_like(case.initial[..., :6])
                physical[..., 2] = case.coords[0]
                state = case.pml.pack(physical)
                result = case.pml.physical(case.rhs(state, 0.))
                # Away from PEC and PML: Ez=x gives dHy/dt=1.
                cell = (10, 7, 10)
                expected = torch.zeros_like(result[cell])
                expected[..., 4] = 1.
                torch.testing.assert_close(result[cell], expected, atol=2e-12, rtol=2e-12)

    def test_rk4_multistep_through_source_ramp_end(self):
        case = ParabolicAntennaCase(refinement=1)
        native = NativeMaxwell(case)
        reference = torch.randn_like(case.initial).masked_fill(case.solid_nodes[..., None], 0.)
        actual = reference.clone()
        for i in range(4):
            t, dt = 1.98+i*.007, .007
            reference = rk4(case.rhs, reference, t, dt)
            actual = native.step(None, actual, t, dt)
        torch.testing.assert_close(actual, reference, rtol=4e-13, atol=4e-12)
        self.assertEqual(actual[case.solid_nodes.expand(case.shape)].abs().max().item(), 0.)

    def test_openmp_matches_serial_and_rejects_wrong_layout(self):
        case = ParabolicAntennaCase(refinement=1)
        state = torch.randn_like(case.initial).masked_fill(case.solid_nodes[..., None], 0.)
        serial = case.native.step(None, state, .3, .005)
        torch.set_num_threads(2)
        parallel = NativeMaxwell(case)
        torch.testing.assert_close(parallel.step(None, state, .3, .005), serial, rtol=0., atol=0.)
        with self.assertRaises(ValueError):
            parallel.rhs(state.float(), .3)
        with self.assertRaises(ValueError):
            parallel.rhs(state.transpose(3, 4), .3)


if __name__ == '__main__':
    unittest.main()
