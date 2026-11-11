"""Taylor–Green initial conditions, conservation, restarts and reference metrics."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from case.taylor_green_vortex import TaylorGreenVortex
from case.taylor_green_vortex.solve import run
from case.taylor_green_vortex.postprocess import uniform_sample, spectrum, load_reference, compare_runs, write_csv


class TaylorGreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initialization_and_periodicity(self):
        for initialization in ('constant-density', 'constant-temperature'):
            m = TaylorGreenVortex((4, 4, 4), initialization=initialization)
            U = m.initial()
            rho, vel, p = m.eq.primitive(U)
            self.assertEqual(m.s.periodic, (True, True, True))
            self.assertEqual(m.forcing, 0.)
            self.assertEqual(m.mu, 1/1600)
            torch.testing.assert_close(vel[..., 2], torch.zeros_like(rho))
            if initialization == 'constant-density':
                torch.testing.assert_close(rho, torch.ones_like(rho), rtol=0., atol=0.)
            else:
                torch.testing.assert_close(p/rho, torch.full_like(rho, m.p0), rtol=2e-15, atol=0.)
            for axis in range(3):
                shifted = list(m.s.coords)
                shifted[axis] = shifted[axis]+2*math.pi
                torch.testing.assert_close(m.initial_at(shifted), U, rtol=3e-14, atol=3e-14)
            d = m.diagnostics(U)
            self.assertAlmostEqual(d['kinetic_energy'], .125, places=12)
        low = TaylorGreenVortex((4, 4, 4), mach=.1)
        high = TaylorGreenVortex((4, 4, 4), mach=.3)
        torch.testing.assert_close(low.eq.primitive(low.initial())[1], high.eq.primitive(high.initial())[1])
        self.assertAlmostEqual(low.p0/high.p0, 9.)

    def test_vorticity_convergence_and_viscous_budget(self):
        errors = []
        for n in (4, 8):
            m = TaylorGreenVortex((n,)*3)
            U = m.initial()
            rho, vel, p, grad, omega, q = m.flow_fields(U)
            x, y, z = m.s.coords
            exact = torch.stack((-torch.cos(x)*torch.sin(y)*torch.sin(z),
                                 -torch.sin(x)*torch.cos(y)*torch.sin(z),
                                 2*torch.sin(x)*torch.sin(y)*torch.cos(z)), -1)
            errors.append(float(m.s.integral((omega-exact).square().sum(-1, keepdim=True))[0].sqrt()))
            viscous = m.ldg.divergence(m.eq.viscous_flux(U, grad))
            d = m.diagnostics(U, rhs=viscous)
            self.assertAlmostEqual(d['epsilon_kinetic'], d['epsilon_viscous'], places=13)
            self.assertLess(abs(d['enstrophy']-.375), 2e-4)
        self.assertLess(errors[1], errors[0]/5)

    def test_native_torch_multistep_and_invariants(self):
        m = TaylorGreenVortex((5, 4, 3))
        native, reference = m.make_stepper('native-p3'), m.make_stepper('torch')
        self.assertFalse(native.constant_mass_flux)
        actual = expected = m.initial()
        total = m.s.integral(actual)
        torch.testing.assert_close(native.plan(actual), m.rhs(actual), rtol=1e-11, atol=1e-11)
        dt = m.timestep(actual, .3)
        for step in range(8):
            actual = native(actual, step*dt, dt)
            expected = reference(expected, step*dt, dt)
            torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-12)
        torch.testing.assert_close(m.s.integral(actual), total, rtol=2e-14, atol=2e-11)
        self.assertEqual(native.last_force, 0.)

    def test_sampling_polynomial_and_fft_parseval(self):
        m = TaylorGreenVortex((3, 2, 4))
        x, y, z = m.s.coords
        sample = uniform_sample(m, (x*x+y*z)[..., None])
        coords = [(torch.arange(n*4, dtype=torch.float64)+.5)*(2*math.pi/(n*4))-math.pi for n in m.s.cells]
        xx, yy, zz = torch.meshgrid(*coords, indexing='ij')
        torch.testing.assert_close(sample[..., 0], xx*xx+yy*zz, rtol=2e-14, atol=2e-14)
        velocity = torch.stack((torch.sin(xx)*torch.cos(yy)*torch.cos(zz),
                                -torch.cos(xx)*torch.sin(yy)*torch.cos(zz), torch.zeros_like(xx)), -1)
        density = 1+.02*torch.cos(xx)
        rows = spectrum(velocity, density)
        self.assertAlmostEqual(sum(r['velocity_energy'] for r in rows), .125, places=14)
        self.assertAlmostEqual(sum(r['density_weighted_energy'] for r in rows),
                               float((.5*density*velocity.square().sum(-1)).mean()), places=14)
        self.assertAlmostEqual(rows[2]['velocity_energy'], .125, places=14)
        self.assertEqual(sum(r['mode_count'] for r in rows), math.prod(velocity.shape[:3]))

    def test_restart_and_snapshot_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'direct', Path(tmp)/'restart'
            args = dict(cells=(3, 3, 3), final_time=.08, cfl=.3, diagnostic_interval=2,
                        checkpoint_interval=2, snapshot_times=(), fields=False, plot=False)
            direct, _ = run(a, **args)
            run(b, **args, max_steps=3)
            resumed, _ = run(b, **args, resume=True)
            torch.testing.assert_close(direct, resumed, rtol=0., atol=0.)
            self.assertEqual(json.loads((b/'status.json').read_text())['status'], 'complete')
            with self.assertRaises(ValueError):
                run(b, **args, resume=True, mach=.1)
            with self.assertRaises(ValueError):
                run(b, **args)
            out = Path(tmp)/'snapshots'
            run(out, cells=(2, 2, 2), final_time=.01, diagnostic_interval=1,
                snapshot_times=(0., .01), fields=True, plot=False)
            self.assertEqual(len(list(out.glob('*.spectrum.csv'))), 2)
            self.assertEqual(len(list(out.glob('*.vtk'))), 2)
            for f in out.glob('*.sampling.json'):
                v = json.loads(f.read_text())
                self.assertAlmostEqual(v['density_weighted_spectral_sum'], v['sampled_kinetic_energy'], places=14)

    def test_statistics_frequency_and_checkpoint_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            options = dict(cells=(3, 3, 3), final_time=.08, cfl=.3, checkpoint_interval=2,
                           snapshot_times=(), fields=False, plot=False)
            dense, history = run(Path(tmp)/'dense', **options, diagnostic_interval=1)
            sparse, _ = run(Path(tmp)/'sparse', **options, diagnostic_interval=100)
            torch.testing.assert_close(dense, sparse, rtol=0., atol=0.)
            self.assertEqual(len(list((Path(tmp)/'dense').glob('*.vtk'))), 0)
            self.assertEqual(len(list((Path(tmp)/'dense').glob('*.spectrum.csv'))), 0)
            self.assertTrue((Path(tmp)/'dense'/'checkpoint-00000002.pt').exists())
            for name in ('epsilon_velocity_gradient', 'epsilon_dilatation', 'max_vorticity',
                         'mean_temperature_normalized', 'mean_entropy', 'uprime'):
                self.assertIn(name, history[-1])

    def test_reference_and_identical_comparison(self):
        reference = load_reference()
        self.assertEqual(reference[0]['Ekin comp'], .125)
        self.assertAlmostEqual(reference[0]['ED_S'], .75/1600, places=14)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)/'identity'
            run_dir.mkdir()
            config = dict(reynolds=1600., gamma=1.4, prandtl=.71, mach=.1,
                          initialization='constant-density', p=7)
            (run_dir/'run.json').write_text(json.dumps(dict(config=config)))
            rows = []
            ref_times = np.array([row['Time'] for row in reference])
            rates = -np.gradient(np.array([row['Ekin comp'] for row in reference]), ref_times, edge_order=2)
            for index in range(0, len(reference), 20):
                row = reference[index]
                rows.append(dict(time=row['Time'], kinetic_energy=row['Ekin comp'],
                                 epsilon_strain=row['DR_S'], epsilon_vorticity=row['ED_S'],
                                 enstrophy=row['Enstrophy comp'], epsilon_viscous=row['DR_Sd'],
                                 pressure_dilatation=-row['DR_p'], mass=1., total_energy=20.,
                                 kinetic_energy_unweighted=row['Ekin incomp'],
                                 epsilon_velocity_gradient=row['DR_u'], epsilon_dilatation=row['ED_D'],
                                 max_vorticity=row['Maximum Vorticity'], uprime=row['uprime'],
                                 mean_temperature_normalized=row['Mean Temperature']/reference[0]['Mean Temperature'],
                                 mean_entropy=row['Mean Entropy'], epsilon_kinetic=float(rates[index])))
            write_csv(run_dir/'history.csv', rows)
            result = compare_runs([run_dir], Path(tmp)/'comparison', plot=False)[0]
            self.assertFalse(result['mach_mismatch'])
            self.assertEqual(len(result['comparisons']), 15)
            for metric in result['comparisons'].values():
                self.assertLess(metric['relative_l2'], 1e-12)


if __name__ == '__main__':
    unittest.main()
