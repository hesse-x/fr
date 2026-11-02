import math
import unittest

import numpy as np
import torch

from case.parabolic_antenna.solve import ParabolicAntennaCase
from case.parabolic_antenna.voltage import voltage_signal
from fr.core import CartesianFR
from fr.plotting import FieldSampler


class VoltageLocalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def case(self, **kwargs):
        return ParabolicAntennaCase(diameter=2., focal_length=.5, refinement=2,
            local_refinement=2, p=2, thickness=.25, feed='voltage', domain_half_width=1.5,
            outer_boundary='silver-muller', **kwargs)

    @staticmethod
    def voltage(case, state):
        physical = case.pml.physical(state).reshape(-1, 6)
        return torch.dot(case.feed_coeff, physical[case.feed_ids, 2]).item()

    def test_line_integral_polynomial_and_riesz_lift(self):
        case = self.case()
        field = torch.zeros_like(case.initial[..., :6])
        field[..., 2] = 1+case.coords[2]**2
        value = self.voltage(case, case.pml.pack(field))
        length = case.voltage_length
        self.assertAlmostEqual(value, length+length**3/12, places=13)
        self.assertAlmostEqual(torch.dot(case.feed_coeff, case.feed_lift).item(), 1., places=13)
        # The constrained edge is disjoint from the conductor.
        elements = case.feed_ids//(case.p+1)**3
        self.assertFalse(bool(case.solid.flatten()[elements].any()))

    def test_complete_mesh_edge_and_no_impressed_volume_current(self):
        case = self.case()
        nodes = (case.p+1)**3
        edge_cells = (case.feed_ids//nodes)%case.cells[2]
        self.assertEqual(len(edge_cells.unique()), 1)
        k = int(edge_cells[0])
        self.assertAlmostEqual(float(case.edges[2][k]), -case.voltage_length/2, places=13)
        self.assertAlmostEqual(float(case.edges[2][k+1]), case.voltage_length/2, places=13)
        self.assertEqual(case.drive_cos.abs().max().item(), 0.)

    def test_homogeneous_voltage_constraint_has_no_spurious_energy_input(self):
        case = self.case(voltage_amplitude=0.)
        torch.manual_seed(184)
        physical = torch.randn_like(case.initial[..., :6]).masked_fill(case.solid_nodes[..., None], 0.)
        flat = physical.reshape(-1, 6)
        voltage = torch.dot(case.feed_coeff, flat[case.feed_ids, 2])
        flat[case.feed_ids, 2] -= voltage*case.feed_lift
        state = case.pml.pack(physical)
        derivative = case.pml.physical(case.rhs(state, 0.))
        energy_rate = case.integral((physical*derivative).sum(-1, keepdim=True)).item()
        self.assertLess(energy_rate, 0.)  # PEC/upwind/ABC, and zero port voltage.
        self.assertAlmostEqual(self.voltage(case, case.rhs(state, 0.)), 0., places=10)

    def test_voltage_and_derivative_enforced_at_accepted_states(self):
        case = self.case()
        state = case.initial.clone()
        t = 0.
        for dt in (.003, .004, .002):
            state = case.native.step(None, state, t, dt)
            t += dt
            self.assertAlmostEqual(self.voltage(case, state), voltage_signal(t)[0], places=12)
            self.assertAlmostEqual(self.voltage(case, case.rhs(state, t)), voltage_signal(t)[1], places=10)
            self.assertTrue(bool(torch.isfinite(state).all()))
        for t in (1.997, 2.003, 2.25):
            state = case.native.step(None, state, t, .001)
            self.assertAlmostEqual(self.voltage(case, state), voltage_signal(t+.001)[0], places=11)

    def test_native_nonuniform_derivatives_against_common_fr(self):
        # Remove all solids and the voltage functional from a fresh native
        # plan. This checks real nonuniform metric factors against the common
        # FR operator, without retaining another antenna backend.
        case = self.case()
        case.solid.zero_()
        case.feed_ids = torch.empty(0, dtype=torch.int64)
        case.feed_coeff = case.feed_lift = torch.empty(0, dtype=torch.float64)
        from case.parabolic_antenna.native import NativeMaxwell
        native = NativeMaxwell(case)
        reference = CartesianFR(case.cells, case.bounds, case.p, flux=case.pml.flux,
            numerical_flux=case.pml.numerical_flux(case.equation.roe_flux),
            boundary=lambda coords, inside, axis, side, t: torch.zeros_like(inside), edges=case.edges)
        torch.manual_seed(77)
        state = torch.randn_like(case.initial)
        torch.testing.assert_close(native.rhs(state, 0.), reference.rhs(state, 0.), rtol=5e-13, atol=5e-12)

    def test_closed_surface_traces_known_polynomial(self):
        case = self.case()
        x, y, z = case.coords
        physical = torch.stack((x*x, y*y, z*z, x*y, y*z, x*z), -1)
        values = case.surface(case.pml.pack(physical))
        x, y, z = case.surface_xyz.T
        expected = torch.stack((x*x, y*y, z*z, x*y, y*z, x*z), -1)
        torch.testing.assert_close(values, expected, atol=2e-12, rtol=0.)

    def test_surface_quadrature_and_nonuniform_plot(self):
        case = self.case()
        limits = 1.25
        self.assertAlmostEqual(case.surface_weights.sum().item(), 6*(2*limits)**2, places=11)
        closure = (case.surface_weights[:, None]*case.surface_normals).sum(0)
        self.assertLess(closure.norm().item(), 1e-12)
        sampler = FieldSampler(case)
        sample = sampler.state((case.coords[0]+2*case.coords[1]+3*case.coords[2])[..., None])[:, 0].numpy()
        centers = sampler.polygons.mean(1)
        np.testing.assert_allclose(sample, centers[:, 0]+2*centers[:, 1], atol=2e-12, rtol=0.)
        self.assertLess(min(float(w.min()) for w in case.widths), max(float(w.max()) for w in case.widths))


if __name__ == '__main__':
    unittest.main()
