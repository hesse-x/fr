import math
import unittest

import torch

from case.uniform_array.solve import PHASE_PRESETS, UniformArrayCase, array_factor, far_field


class UniformArrayTests(unittest.TestCase):
    def test_closed_surface_transform_against_exact_dipole(self):
        # Independent analytic electric dipole, including its reactive near
        # field: catches current orientation, phasor sign and area mistakes.
        case = UniformArrayCase(elements=1, refinement=2, p=4)
        x = case.surface_xyz
        r = x.norm(dim=-1, keepdim=True)
        n = x/r
        p = torch.tensor([0., 0., 1.], dtype=torch.float64).expand_as(n)
        transverse = p-n*(n*p).sum(-1, keepdim=True)
        k = 2*math.pi
        factor = torch.exp(1j*k*r)/(4*math.pi)
        electric = factor*(k*k/r*transverse+(3*n*(n*p).sum(-1, keepdim=True)-p)*(1/r**3-1j*k/r**2))
        magnetic = factor*(k*k/r+1j*k/r**2)*torch.linalg.cross(n, p)
        directions = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.6, 0., .8]], dtype=torch.float64)
        actual = far_field(x, case.surface_normals, case.surface_weights,
                           torch.cat((electric, magnetic), -1), directions)
        expected = k*k/(4*math.pi)*(p[:len(directions)]-directions*directions[:, 2:])
        self.assertLess((actual-expected).norm().item()/expected.norm().item(), 1e-6)

    def test_surface_area_and_normal_closure(self):
        case = UniformArrayCase()
        self.assertAlmostEqual(case.surface_weights.sum().item(), 14.)
        closure = (case.surface_weights[:, None]*case.surface_normals).sum(0)
        self.assertLess(closure.norm().item(), 1e-12)

    def test_steering_phase(self):
        positions = torch.arange(4, dtype=torch.float64)*.5
        sine = .5
        phase = 2*math.pi*positions*sine
        directions = torch.tensor([[sine, math.sqrt(1-sine*sine), 0.], [0., 1., 0.]], dtype=torch.float64)
        af = array_factor(directions, positions, phase)
        self.assertAlmostEqual(af[0].abs().item(), 4.)
        self.assertLess(af[1].abs().item(), 1e-12)

    def test_invalid_geometry(self):
        for kwargs in ({'elements': 0}, {'spacing': 0}, {'radius': .5}, {'steer': 91}, {'refinement': 0}):
            with self.assertRaises(ValueError):
                UniformArrayCase(**kwargs)

    def test_phase_presets_radiation_and_source_sign(self):
        theta = torch.linspace(-90., 90., 361, dtype=torch.float64)
        a = theta*math.pi/180
        directions = torch.stack((a.sin(), a.cos(), torch.zeros_like(a)), -1)
        for name, expected in (('scan30', 30.), ('scan_minus30', -30.), ('endfire', -90.)):
            case = UniformArrayCase(phases=PHASE_PRESETS[name])
            power = array_factor(directions, case.positions, case.phase).abs().square()
            self.assertAlmostEqual(theta[power.argmax()].item(), expected)
        case = UniformArrayCase(phases=PHASE_PRESETS['difference'])
        af = array_factor(directions, case.positions, case.phase)
        self.assertLess(af[180].abs().item(), 1e-12)
        # Global phase rotates the time signal without changing its pattern.
        shifted = array_factor(directions, case.positions, case.phase+.37)
        torch.testing.assert_close(shifted, af*complex(math.cos(.37), math.sin(.37)))
        # Physical source integrates to minus the total complex excitation.
        # At t=2 and 2.25 the cosine and sine components are sampled separately.
        case = UniformArrayCase(phases=(0., 20., 40., 60.))
        for t, expected in ((2., -case.phase.cos().sum()), (2.25, -case.phase.sin().sum())):
            source = case.pml.physical(case.driving_source(case.coords, case.initial, t))
            self.assertAlmostEqual(case.integral(source)[2].item(), expected.item(), places=11)

    def test_invalid_explicit_phases(self):
        for kwargs in ({'phases': [0., 90.]}, {'phases': [0., 0., 0., float('nan')]},
                       {'phases': [0., 0., 0., 0.], 'steer': 30.}):
            with self.assertRaises(ValueError):
                UniformArrayCase(**kwargs)


if __name__ == '__main__':
    unittest.main()
