import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from case.round_jet_les.generate_mesh import build_mesh
from case.round_jet_les.postprocess import FieldStatistics, Probes
from case.round_jet_les.solve import RoundJetLES, run


class RoundJetLESTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        mesh = build_mesh((4, 4, 4), length=4., half_width=1.5)
        self.model = RoundJetLES(mesh)

    def test_quiescent_state_and_starting_jet(self):
        model = self.model
        ambient = model.initial()
        residual = model.rhs(ambient, 0.)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0.)
        self.assertEqual(model.s.p, 3)
        self.assertEqual(model.sgs_model, 'wale')
        running = model.rhs(ambient, .5)
        self.assertGreater(float(running[..., 1].max()), 0.)
        self.assertTrue(torch.isfinite(running).all())

    def test_native_open_faces_and_rk_match_torch(self):
        model = self.model
        x, y, z = model.s.coords
        velocity = torch.stack((.2+.1*y.sin(), .05*z.cos(), .04*x.sin()), -1)
        state = model.soa(model.eq.conservative(1.+.02*x.cos(), velocity, model.pressure+.01*y.sin()))
        native = model.make_stepper('native-p3', block=(2, 2, 2))
        t = .7
        rhs = model.rhs(state, t)
        torch.testing.assert_close(native.plan(state, t), rhs, atol=2e-11, rtol=1e-10)
        other = model.make_stepper('native-p3', block=(2, 2, 2), boundary_mode='fused')
        torch.testing.assert_close(other.plan(state, t), rhs, atol=2e-11, rtol=1e-10)
        convection = native.plan.convection(state, t)
        torch.testing.assert_close(native.plan.viscous(state, convection, t), rhs, atol=2e-11, rtol=1e-10)
        valid, dt = native.scan(state, 1.6, t=t)
        self.assertTrue(valid)
        self.assertAlmostEqual(dt, model.timestep(state, 1.6, t=t), places=12)
        actual, valid, next_dt = native.advance_finish(state, t, dt, 1.6)
        expected = model.make_stepper('torch')(state, t, dt)
        self.assertTrue(valid)
        self.assertAlmostEqual(next_dt, model.timestep(actual, 1.6, t=t+dt), places=12)
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=1e-10)

    def test_dynamic_packets_match_static(self):
        torch.set_num_threads(3)
        model = self.model
        static = model.make_stepper('native-p3', block=(2, 2, 2))
        dynamic = model.make_stepper('native-p3', block=(2, 2, 2), group_packet=2)
        state = model.soa(model.initial())
        t = .7
        torch.testing.assert_close(dynamic.plan(state, t), static.plan(state, t), atol=0., rtol=0.)
        _, dt = static.scan(state, 1.6, t=t)
        expected, valid, next_dt = static.advance_finish(state, t, dt, 1.6)
        actual, dynamic_valid, dynamic_dt = dynamic.advance_finish(state, t, dt, 1.6)
        self.assertTrue(valid and dynamic_valid)
        self.assertEqual(next_dt, dynamic_dt)
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)

    def test_acoustic_outlet_native_matches_reference_with_backflow(self):
        model = RoundJetLES(build_mesh((4, 4, 4), length=4., half_width=1.5), outlet='nscbc')
        x, y, z = model.s.coords
        velocity = torch.stack((.45*y+.1*x.sin(), .05*z.cos(), .04*x.sin()), -1)
        state = model.soa(model.eq.conservative(1.+.02*x.cos(), velocity,
                                                model.pressure+.01*x.sin()*y.cos()))
        native = model.make_stepper('native-p3', group_packet=2)
        torch.testing.assert_close(native.plan(state, .7), model.rhs(state, .7),
                                   atol=2e-11, rtol=1e-10)
        ambient = model.soa(model.initial())
        torch.testing.assert_close(native.plan(ambient, 0.), torch.zeros_like(ambient),
                                   atol=2e-11, rtol=0.)

    def test_recycled_output_preserves_owned_states(self):
        model = self.model
        modal_filter = model.modal_filter
        for enabled in (True, False):
            with self.subTest(filter=enabled):
                model.modal_filter = modal_filter if enabled else None
                stepper = model.make_stepper('native-p3', group_packet=2)
                initial = model.soa(model.initial())
                state, out = initial.clone(), torch.empty_like(initial)
                t = .4
                _, dt = stepper.scan(initial, 1.6, t=t)
                moments = initial.new_zeros((12, *model.s.shape)).movedim(0, -1)
                recycled_moments = torch.zeros_like(moments)
                owned, valid, next_dt = stepper.advance_finish(initial, t, dt, 1.6, moments)
                saved = owned.clone()
                actual, recycled_valid, recycled_dt = stepper.advance_finish(
                    state, t, dt, 1.6, recycled_moments, out=out)
                self.assertIs(actual, out)
                self.assertTrue(valid and recycled_valid)
                self.assertEqual(next_dt, recycled_dt)
                self.assertTrue(torch.equal(owned, out))
                self.assertTrue(torch.equal(moments, recycled_moments))
                stepper.advance_finish(out, t+dt, next_dt, 1.6, out=state)
                self.assertTrue(torch.equal(owned, saved))
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    stepper.advance_finish(state, t, dt, 1.6, out=state)

    def test_probe_reconstructs_polynomial_on_stretched_grid(self):
        s = self.model.s
        x, y, z = s.coords
        coordinates = tuple(s.z.new_tensor(points) for points in (
            (0., .9, 4.), (-1.5, -.3, .8), (0., 1.2, -.2)))
        probe = Probes(s, coordinates)
        a, b, c = coordinates
        torch.testing.assert_close(probe(x*x+2*y*z), a*a+2*b*c, atol=1e-12, rtol=0.)

    def test_run_writes_time_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            state = run(self.model, path, final_time=.04, cfl=1.6, sample_from=.02,
                        sample_interval=.01, plot=False)
            self.assertTrue(self.model.eq.valid(state))
            summary = json.loads((path/'summary.json').read_text())
            self.assertEqual(summary['samples'], 3)
            self.assertEqual(summary['time_Uj_D'], .04)
            self.assertEqual(summary['backend'], 'native-p3')
            self.assertEqual(summary['group_packet'], 2)
            with (path/'centerline.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 161)

    def test_checkpoint_resume_preserves_state_and_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline, restarted = Path(tmp)/'baseline', Path(tmp)/'restarted'
            options = dict(final_time=.08, cfl=1.6, sample_from=.02, sample_interval=.01, plot=False)
            state = run(self.model, baseline, **options)
            run(self.model, restarted, max_steps=2, checkpoint_every=1, **options)
            partial = json.loads((restarted/'summary.json').read_text())
            self.assertGreater(partial['samples'], 0)
            self.assertLess(partial['time_Uj_D'], options['final_time'])
            resumed = run(self.model, restarted, resume=True, **options)
            torch.testing.assert_close(resumed, state, atol=0., rtol=0.)
            for file in ('centerline.csv', 'radial_profiles.csv'):
                self.assertEqual((baseline/file).read_bytes(), (restarted/file).read_bytes())
            with self.assertRaisesRegex(ValueError, 'sampling configuration'):
                run(self.model, restarted, resume=True, **dict(options, sample_interval=.02))

    def test_checkpoint_allows_a_new_sampling_start_before_first_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            options = dict(final_time=.04, cfl=1.6, sample_interval=.01, plot=False)
            run(self.model, path, max_steps=1, checkpoint_every=1, sample_from=.03, **options)
            run(self.model, path, resume=True, sample_from=.02, **options)
            summary = json.loads((path/'summary.json').read_text())
            self.assertEqual(summary['samples'], 3)

    def test_reset_samples_and_resume_across_sampling_start_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            options = dict(final_time=.15, cfl=1.6, sample_interval=.01, plot=False)
            run(self.model, path, max_steps=3, checkpoint_every=1, sample_from=.015, **options)
            self.assertGreater(json.loads((path/'summary.json').read_text())['samples'], 0)

            resumed = dict(options, sample_from=None, sample_from_step=5, resume=True)
            run(self.model, path, max_steps=4, reset_samples=True, **resumed)
            pending = torch.load(path/'checkpoint.pt', weights_only=True)
            self.assertEqual(pending['sample_count'], 0)
            self.assertEqual(torch.count_nonzero(pending['sample_sum']).item(), 0)
            self.assertIsNone(pending['first_sample_time'])

            resumed['full_field_statistics'] = True
            run(self.model, path, max_steps=5, **resumed)
            first = torch.load(path/'checkpoint.pt', weights_only=True)
            self.assertEqual(first['sample_count'], 1)
            self.assertEqual(first['first_sample_time'], first['time'])
            self.assertEqual(torch.count_nonzero(first['field_moments'][1]).item(), 0)
            run(self.model, path, **resumed)
            summary = json.loads((path/'summary.json').read_text())
            self.assertGreater(summary['samples'], 1)
            self.assertEqual(summary['sample_from'], first['time'])
            self.assertEqual(summary['sample_from_step'], 5)
            field = torch.load(path/summary['field_statistics'], weights_only=True)
            self.assertEqual(field['samples'], summary['samples'])
            self.assertEqual(field['velocity_mean_Uj'].shape, (3, *self.model.s.shape))
            torch.testing.assert_close(field['resolved_tke_Uj2'],
                                       .5*field['reynolds_stress_Uj2'][:3].sum(0))

    def test_field_statistics_matches_direct_velocity_covariance(self):
        model = self.model
        density = model.s.coords[0].new_ones(model.s.shape)
        values = density.new_tensor(((1., 2., 3.), (3., -1., 0.), (2., 2., -1.)))
        samples = FieldStatistics(model)
        weights = density.new_tensor([1., 2., 3.])
        total_weight = 0.
        for value, weight in zip(values, weights.tolist()):
            velocity = value.expand(*density.shape, 3)
            total_weight += weight
            samples.add(model.eq.conservative(density, velocity, model.pressure), weight, total_weight)
        mean, covariance_sum = samples.moments
        index = (slice(None),)+(0,)*density.ndim
        expected_mean = (values*weights[:, None]).sum(0)/total_weight
        delta = values-expected_mean
        expected = torch.stack([(delta[:, i]*delta[:, j]*weights).sum()/total_weight
                                for i, j in samples.stress_pairs])
        torch.testing.assert_close(mean[index][1:4], expected_mean)
        torch.testing.assert_close(covariance_sum[index]/total_weight, expected)

    def test_time_sampling_caps_step_gap_and_resumes_between_samples(self):
        for interval in (.018, .05):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as tmp:
                baseline, resumed = Path(tmp)/'baseline', Path(tmp)/'resumed'
                options = dict(final_time=.3, cfl=1.6, sample_from=None, sample_interval=interval,
                               sample_from_step=2, sample_max_steps=2, plot=False,
                               checkpoint_every=2, full_field_statistics=True)
                run(self.model, baseline, max_steps=10, **options)
                run(self.model, resumed, max_steps=7, **options)
                partial = torch.load(resumed/'checkpoint.pt', weights_only=True)
                self.assertEqual(partial['last_sample_step'], 6)
                self.assertEqual(partial['sample_count'], 2)
                if interval == .018:
                    self.assertAlmostEqual(partial['sample_weight'], 2*interval)
                else:
                    self.assertLess(partial['sample_weight'], 2*interval)
                run(self.model, resumed, max_steps=10, resume=True, **options)
                expected = torch.load(baseline/'checkpoint.pt', weights_only=True)
                actual = torch.load(resumed/'checkpoint.pt', weights_only=True)
                self.assertEqual(actual['sample_count'], 4)
                self.assertEqual(actual['last_sample_step'], 10)
                self.assertAlmostEqual(actual['sample_weight'], actual['time']-actual['first_sample_time'])
                for key in ('state', 'sample_sum', 'field_moments'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0.)

    def test_every_step_sampling_uses_equal_weights_and_keeps_solver_timestep(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            options = dict(final_time=.3, cfl=1.6, sample_from=None, sample_interval=1e-6,
                           sample_from_step=2, sample_max_steps=1, sample_weighting='step',
                           plot=False, checkpoint_every=2, full_field_statistics=True)
            run(self.model, path, max_steps=5, **options)
            first = torch.load(path/'checkpoint.pt', weights_only=True)
            self.assertEqual(first['sample_count'], 3)
            self.assertEqual(first['sample_weight'], 3.)
            self.assertGreater(first['time']-first['first_sample_time'], .001)

            run(self.model, path, max_steps=8, resume=True, **options)
            resumed = torch.load(path/'checkpoint.pt', weights_only=True)
            summary = json.loads((path/'summary.json').read_text())
            self.assertEqual(resumed['sample_count'], 6)
            self.assertEqual(resumed['sample_weight'], 6.)
            self.assertEqual(summary['sample_weighting'], 'step')
            self.assertAlmostEqual(summary['averaged_time_Uj_D'],
                                   resumed['time']-first['first_sample_time'])


if __name__ == '__main__':
    unittest.main()
