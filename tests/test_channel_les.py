"""Wall-bounded SGS residual, time step and constant-flux native advancement."""
from importlib import import_module
import unittest
import torch

ChannelLES = import_module('case.channel-les').ChannelLES
Channel = import_module('case.channel-dns.channel').Channel


class ChannelLESTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_coarse_mesh_geometry(self):
        from fr import CartesianMesh
        from pathlib import Path
        generator = import_module('case.channel-les.generate_mesh')
        mesh = generator.build_mesh()
        dns = CartesianMesh.read(generator.DNS_MESH)
        self.assertEqual(mesh.cells, (32, 12, 16))
        self.assertEqual(mesh.bounds, dns.bounds)
        # x/z retain the earlier coarsened DNS spacing; y clusters at both walls.
        torch.testing.assert_close(mesh.edges[0], dns.edges[0][::2])
        torch.testing.assert_close(mesh.edges[2], dns.edges[2][::2])
        eta = torch.linspace(-1., 1., 13, dtype=torch.float64)
        expected_y = torch.tanh(1.2*eta)/torch.tanh(torch.tensor(1.2, dtype=torch.float64))
        torch.testing.assert_close(mesh.edges[1], expected_y)
        torch.testing.assert_close(mesh.edges[1], -mesh.edges[1].flip(0))
        widths = [e.diff() for e in mesh.edges]
        self.assertTrue(bool((widths[1][:5].diff() > 0).all()))
        self.assertLess(float(widths[1][0]), float(widths[1][5]))
        torch.testing.assert_close(widths[1], widths[1].flip(0))
        self.assertLess(max(float(w.max()) for w in widths)/min(float(w.min()) for w in widths), 4.6)
        saved = CartesianMesh.read(Path(generator.__file__).with_name('mesh')/'default.json')
        for a, b in zip(mesh.edges, saved.edges):
            torch.testing.assert_close(a, b)
        model = ChannelLES(mesh.cells, mesh=mesh, forcing=0.,
                           lengths=(mesh.bounds[0][1], mesh.bounds[2][1]))
        nearest = 180.*(1+float(model.s.z[0]))*float(widths[1][0])/2
        self.assertTrue(1.08 < nearest < 1.1)
        torch.testing.assert_close(model.s.edges[1], mesh.edges[1])

    def test_fatal_diagnostics_distinguish_invalid_states(self):
        import json
        inspect = import_module('case.channel-dns.long_run').state_failure_details
        model = ChannelLES((1, 2, 1), forcing=0.)
        state = model.initial()
        original = state.clone()
        index = (0, 0, 0, 0, 0, 0)
        state[index][4] = 0.  # Finite density with negative internal energy.
        report = inspect(model, state)
        self.assertEqual(report['counts']['nonpositive_pressure'], 1)
        self.assertEqual(report['counts']['nonpositive_density'], 0)
        self.assertEqual(report['examples']['nonpositive_pressure']['cell'], [0, 0, 0])
        self.assertLess(report['examples']['nonpositive_pressure']['pressure'], 0.)
        state.copy_(original)
        state[index][0] = -1.
        self.assertEqual(inspect(model, state)['counts']['nonpositive_density'], 1)
        state.copy_(original)
        state[index][4] = float('nan')
        report = inspect(model, state)
        self.assertEqual(report['counts']['nonfinite_pressure'], 1)
        self.assertEqual(report['counts']['nonfinite_conservative'], 1)
        json.dumps(report, allow_nan=False)
        state.copy_(original)
        inspect(model, state)
        torch.testing.assert_close(state, original)

    def test_fatal_history_is_saved_without_accepting_bad_step(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from fr.native.cartesian_ns.native_step import NativeLocalRKStepper
        runner = import_module('case.channel-les.solve')
        calls = []
        def advance(native, state, t, dt, cfl, integral=None):
            candidate = state.clone(memory_format=torch.preserve_format)
            calls.append(t)
            if len(calls) == 3:
                candidate[..., 4] = 0.
            return candidate, len(calls) < 3, dt
        with tempfile.TemporaryDirectory() as directory:
            argv = ['channel-les', '--cells', '2', '6', '2', '--steps', '3',
                    '--threads', '2', '--output', directory]
            with patch('sys.argv', argv), patch.object(NativeLocalRKStepper, 'advance_finish', advance), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as log:
                with self.assertRaisesRegex(RuntimeError, 'nonphysical state on step 3'):
                    runner.main()
            self.assertIn('nonpositive_pressure', log.getvalue())
            failure = json.loads((Path(directory)/'failure.json').read_text())
            self.assertEqual(failure['failed_step'], 3)
            self.assertEqual(failure['history_steps'], [0, 1, 2])
            history = torch.load(Path(directory)/'failure_history.pt', weights_only=False)
            self.assertEqual([s['step'] for s in history['history']], [0, 1, 2])
            self.assertTrue(bool((history['history'][-1]['U'][..., 4] > 0).all()))
            self.assertTrue(bool((history['failed_state'][..., 4] == 0).all()))
            self.assertFalse((Path(directory)/'checkpoint.pt').exists())

    def test_wale_limits(self):
        from fr.les import WALE
        closure = WALE()
        g = torch.zeros(4, 3, dtype=torch.float64)
        for shear in (0., 2.):
            g[0, 1] = shear
            self.assertEqual(float(closure.viscosity(torch.tensor(1.), g, .1)), 0.)
        g[:3] = torch.diag(torch.tensor([1., -1., 0.]))
        # S:S=2, Sd:Sd=2/3 for this diagonal gradient.
        expected = .25*.1**2*(2/3)**1.5/(2**2.5+(2/3)**1.25)
        self.assertAlmostEqual(float(closure.viscosity(torch.tensor(1.), g, .1)), expected)
        with self.assertRaises(ValueError):
            WALE(cw=-1.)

    def test_zero_closure_recovers_dns(self):
        dns = Channel((2, 4, 2), forcing=0.)
        les = ChannelLES((2, 4, 2), cw=0., forcing=0.)
        state = dns.initial()
        torch.testing.assert_close(les.rhs(state), dns.rhs(state), rtol=0., atol=0.)

    def test_native_wall_sgs_and_constant_mass_flux(self):
        for closure in ('wale', 'vreman', 'smagorinsky'):
            with self.subTest(closure=closure):
                model = ChannelLES((2, 4, 2), forcing=0., sgs_model=closure)
                state = model.soa(model.initial())
                native = model.make_stepper('native-p3', constant_mass_flux=True)
                torch.testing.assert_close(native.plan(state), model.rhs(state), rtol=1e-10, atol=2e-10)
                valid, dt = native.scan(state, .1)
                self.assertTrue(valid)
                self.assertAlmostEqual(dt, model.timestep(state, .1), places=13)
                advanced, valid, _ = native.advance_finish(state, 0., dt, .1)
                self.assertTrue(valid)
                torch.testing.assert_close(advanced, model.make_stepper('torch', constant_mass_flux=True)(state, 0., dt), rtol=2e-12, atol=2e-10)
                torch.testing.assert_close(model.s.integral(advanced)[1], model.s.integral(state)[1], rtol=2e-13, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
