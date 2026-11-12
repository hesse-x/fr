"""Closure limits, conservative residual, explicit stepping and restart."""
import tempfile
import json
from pathlib import Path
import unittest
import torch
from fr.les import Smagorinsky, Vreman
from case.taylor_green_vortex import TaylorGreenVortex
from case.taylor_green_vortex_les import TaylorGreenVortexLES, run


class TaylorGreenLESTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_closure_shear_and_rigid_rotation(self):
        sgs = Smagorinsky(.1)
        g = torch.zeros(4, 3, dtype=torch.float64)
        g[0, 1] = 2.
        self.assertAlmostEqual(float(sgs.viscosity(torch.tensor(3.), g, .5)), .015)
        from fr.equations.navier_stokes import NavierStokes
        eq = NavierStokes()
        velocity = torch.tensor([1., 0., 0.], dtype=g.dtype)
        state = eq.conservative(torch.tensor(3.), velocity, torch.tensor(1.))
        g[3, 0] = 4.
        flux = sgs.flux(eq, state, g, .5)
        self.assertAlmostEqual(float(flux[1, 1]), .03)
        self.assertAlmostEqual(float(flux[4, 0]), .015*eq.gamma/(eq.gamma-1)/.9*4)
        self.assertAlmostEqual(float(flux[4, 1]), .03)
        torch.testing.assert_close(flux[0], torch.zeros(3, dtype=g.dtype))
        g[1, 0] = -2.
        self.assertEqual(float(sgs.viscosity(torch.tensor(3.), g, .5)), 0.)
        for cs, pr in ((-1., .9), (float('nan'), .9), (.1, 0.)):
            with self.assertRaises(ValueError):
                Smagorinsky(cs, pr)

    def test_smagorinsky_native_and_hifiles_width(self):
        model = TaylorGreenVortexLES((3, 2, 2), sgs_model='smagorinsky', cs=.1,
                                    filter_ratio=2.)
        self.assertAlmostEqual(float(model.delta), 2*(model.s.cell_width(0, variables=False)*
            model.s.cell_width(1, variables=False)*model.s.cell_width(2, variables=False))**(1/3)/4)
        state = model.initial()
        state = model.soa(state*(1+.02*torch.sin(model.s.coords[0]))[..., None])
        native = model.make_stepper('native-p3')
        torch.testing.assert_close(native.plan(state), model.rhs(state), rtol=1e-10, atol=2e-12)
        valid, dt = native.scan(state, .3)
        self.assertTrue(valid)
        self.assertAlmostEqual(dt, model.timestep(state, .3), places=14)
        advanced, valid, next_dt = native.advance_finish(state, 0., dt, .3)
        self.assertTrue(valid)
        torch.testing.assert_close(advanced, model.make_stepper()(state, 0., dt), rtol=2e-13, atol=2e-12)
        self.assertAlmostEqual(next_dt, model.timestep(advanced, .3), places=14)
        gradient = torch.zeros(4, 3, dtype=torch.float64)
        gradient[:3] = torch.eye(3, dtype=gradient.dtype)
        self.assertEqual(float(model.sgs.viscosity(torch.tensor(1.), gradient, .1)), 0.)

    def test_zero_model_and_conservation(self):
        base = TaylorGreenVortex((5, 4, 3))
        zero = TaylorGreenVortexLES((5, 4, 3), cv=0.)
        les = TaylorGreenVortexLES((5, 4, 3))
        U = base.initial()
        torch.testing.assert_close(zero.rhs(U), base.rhs(U), rtol=0, atol=0)
        torch.testing.assert_close(les.s.integral(les.rhs(U)), torch.zeros(5, dtype=U.dtype), atol=2e-12, rtol=0)
        d = les.diagnostics(U)
        self.assertGreater(d['epsilon_sgs'], 0)
        self.assertGreater(d['mu_sgs_mean'], 0)
        self.assertLessEqual(les.timestep(U), base.timestep(U))
        dt = les.timestep(U, .1)
        advanced = les.make_stepper()(U, 0., dt)
        self.assertTrue(les.eq.valid(advanced))
        torch.testing.assert_close(les.s.integral(advanced), les.s.integral(U), atol=2e-11, rtol=1e-13)
        native = les.make_stepper('native-p3')
        torch.testing.assert_close(native.plan(U), les.rhs(U), rtol=1e-10, atol=2e-12)
        torch.testing.assert_close(native.plan.viscous(U, les.convection(U)), les.rhs(U),
                                   rtol=1e-10, atol=2e-12)
        actual = native(U, 0., dt)
        torch.testing.assert_close(actual, advanced, rtol=2e-13, atol=2e-12)
        valid, native_dt = native.scan(U, .1)
        self.assertTrue(valid)
        self.assertEqual(native_dt, dt)
        actual, valid, next_dt = native.advance_finish(U, 0., dt, .1)
        self.assertTrue(valid)
        torch.testing.assert_close(actual, advanced, rtol=2e-13, atol=2e-12)
        self.assertAlmostEqual(next_dt, les.timestep(actual, .1))

    def test_vreman_invariants(self):
        model = Vreman()
        g = torch.zeros(4, 3, dtype=torch.float64)
        self.assertEqual(float(model.viscosity(torch.tensor(1.), g, .1)), 0.)
        g[0, 1] = 2.
        self.assertEqual(float(model.viscosity(torch.tensor(1.), g, .1)), 0.)
        g[:3] = torch.eye(3, dtype=g.dtype)
        self.assertAlmostEqual(float(model.viscosity(torch.tensor(2.), g, .5)), 2*.07*.25)
        g[:3] = torch.tensor([[1., 2., 3.], [0., -1., 4.], [5., 2., 0.]], dtype=g.dtype)
        du = g[:3]
        beta = .5**2*(du@du.T)
        b = sum(beta[i,i]*beta[j,j]-beta[i,j]**2 for i,j in ((0,1),(0,2),(1,2)))
        expected = 2*.07*(b/du.square().sum()).sqrt()
        torch.testing.assert_close(model.viscosity(torch.tensor(2.), g, .5), expected)
        q, _ = torch.linalg.qr(torch.tensor([[1.,2.,0.],[2.,0.,3.],[0.,3.,1.]], dtype=g.dtype))
        rotated = g.clone()
        rotated[:3] = q@du@q.T
        torch.testing.assert_close(model.viscosity(torch.tensor(2.), rotated, .5), expected)
        for cv in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                Vreman(cv)

    def test_native_diffusion_scan(self):
        # Force the thermal SGS restriction to dominate the acoustic bound.
        les = TaylorGreenVortexLES((5, 3, 2), cv=3., turbulent_prandtl=.3)
        U = les.initial()
        U = les.soa(U*(1+.02*torch.sin(les.s.coords[0]))[..., None])
        native = les.make_stepper('native-p3')
        gradient = les.viscous_operator.gradient(les.primitive(U))
        mu_t = les.sgs.viscosity(U[..., 0], gradient, les.delta)
        expected = max(float((1+mu_t/les.mu).max()),
                       float((1+mu_t/les.mu*les.eq.prandtl/.3).max()))
        self.assertGreater(expected, 10.)
        self.assertAlmostEqual(native.plan.viscous.diffusion_ratio(U), expected, places=10)
        valid, dt = native.scan(U, .3)
        self.assertTrue(valid)
        self.assertAlmostEqual(dt, les.timestep(U, .3), places=14)
        advanced, valid, next_dt = native.advance_finish(U, 0., dt, .3)
        self.assertTrue(valid)
        self.assertAlmostEqual(next_dt, les.timestep(advanced, .3), places=14)
        fused = les.make_stepper('native-p3', boundary_mode='fused')
        self.assertAlmostEqual(fused.plan.viscous.diffusion_ratio(U), expected, places=10)
        torch.testing.assert_close(fused.plan(U), les.rhs(U), atol=2e-12, rtol=1e-10)

    def test_none_and_runtime_coefficients(self):
        base = TaylorGreenVortex((3,)*3)
        none = TaylorGreenVortexLES((3,)*3, sgs_model='none')
        state = none.initial()
        self.assertIsNone(none.sgs)
        torch.testing.assert_close(none.rhs(state), base.rhs(state), rtol=0, atol=0)
        self.assertEqual(none.diagnostics(state)['epsilon_sgs'], 0.)
        plan = none.make_stepper('native-p3').plan
        torch.testing.assert_close(plan(state), base.rhs(state), atol=2e-12, rtol=1e-10)
        self.assertEqual(plan.metadata['sgs_model'], 'none')
        first = TaylorGreenVortexLES((3,)*3, cv=.07).make_stepper('native-p3').plan
        second_model = TaylorGreenVortexLES((3,)*3, cv=.14, turbulent_prandtl=.6)
        second = second_model.make_stepper('native-p3').plan
        self.assertIs(first.library, second.library)
        self.assertIs(first.viscous.library, second.viscous.library)
        torch.testing.assert_close(second(state), second_model.rhs(state), atol=2e-12, rtol=1e-10)
        with self.assertRaises(ValueError):
            TaylorGreenVortexLES((2,)*3, sgs_model='unknown')

    def test_original_convection_defaults(self):
        default = TaylorGreenVortexLES((3, 2, 2))
        explicit = TaylorGreenVortexLES((3, 2, 2), interface_flux='rusanov', anti_alias='none')
        self.assertEqual(default.interface_flux, 'rusanov')
        self.assertFalse(default.s.anti_alias)
        state = default.initial()
        torch.testing.assert_close(default.rhs(state), explicit.rhs(state), atol=0, rtol=0)
        native = default.make_stepper('native-p3')
        reference = explicit.make_stepper('native-p3')
        self.assertIs(native.plan.library, reference.plan.library)
        self.assertEqual(native.plan.metadata['interface_flux'], 'rusanov')
        self.assertEqual(native.plan.metadata['anti_alias'], [])
        with tempfile.TemporaryDirectory() as temporary:
            _, _ = run(temporary, cells=(2,)*3, p=2, backend='torch', final_time=0.,
                       snapshot_times=(), fields=False, plot=False)
            config = json.loads((Path(temporary)/'run.json').read_text())['config']
            self.assertEqual(config.get('interface_flux', 'rusanov'), 'rusanov')
            self.assertEqual(config['anti_alias'], 'none')

    def test_restart(self):
        with tempfile.TemporaryDirectory() as output:
            options = dict(cells=(2,)*3, p=2, final_time=.002, max_dt=.001,
                           backend='torch', snapshot_times=(), fields=False, plot=False, diagnostic_interval=1)
            run(output, max_steps=1, **options)
            U, history = run(output, resume=True, **options)
            self.assertEqual(history[-1]['time'], .002)
            self.assertIn('epsilon_sgs', history[-1])
            comparison = json.loads((Path(output)/'comparison/comparison.json').read_text())['runs'][0]
            metrics = comparison['comparisons']
            self.assertAlmostEqual(metrics['epsilon_modelled_viscous']['ours_peak'],
                                   max(r['epsilon_viscous']+r['epsilon_sgs'] for r in history))
            self.assertAlmostEqual(metrics['epsilon_modelled_physical']['ours_peak'],
                                   max(r['epsilon_viscous']+r['epsilon_sgs']-r['pressure_dilatation'] for r in history))
            self.assertAlmostEqual(metrics['epsilon_viscous']['ours_peak'],
                                   max(r['epsilon_viscous'] for r in history))
            self.assertIn('resolved fields', comparison['les_note'])
            with tempfile.TemporaryDirectory() as uninterrupted:
                reference, _ = run(uninterrupted, **options)
                torch.testing.assert_close(U, reference, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                run(output, resume=True, cv=.2, **options)
