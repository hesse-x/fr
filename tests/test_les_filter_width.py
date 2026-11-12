"""Filter-width conventions must agree in Torch, native fluxes and CFL scans."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import torch
from case.taylor_green_vortex_les import TaylorGreenVortexLES, run


class LESFilterWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_center_distance_and_runtime_scaling(self):
        base = TaylorGreenVortexLES((4,)*3)
        center = TaylorGreenVortexLES((4,)*3, filter_width='gauss-center')
        expected = math.sqrt((15-2*math.sqrt(30))/35)
        ratio = 2*expected
        self.assertAlmostEqual(center.sgs_filter_scale, ratio, places=14)
        self.assertAlmostEqual(center.delta, expected/2*(2*math.pi/4), places=14)
        self.assertEqual(center.sgs.cv, .07)
        state = base.initial()
        gradient = base.viscous_operator.gradient(base.primitive(state))
        old = base.sgs.viscosity(state[..., 0], gradient, base.delta)
        new = center.sgs.viscosity(state[..., 0], gradient, center.delta)
        torch.testing.assert_close(new, old*ratio**2, rtol=3e-15, atol=1e-18)
        a, b = base.make_stepper('native-p3'), center.make_stepper('native-p3')
        self.assertIs(a.plan.library, b.plan.library)
        self.assertIs(a.plan.viscous.library, b.plan.viscous.library)
        torch.testing.assert_close(a.plan.convection(state), b.plan.convection(state), rtol=0, atol=0)
        self.assertEqual(b.plan.metadata['sgs_filter_width'], 'gauss-center')
        self.assertAlmostEqual(b.plan.metadata['sgs_filter_scale'], ratio)

    def test_native_flux_step_and_diffusion_scan(self):
        for mode in ('gauss-center', 'half-dof'):
            model = TaylorGreenVortexLES((5,3,2), cv=3., turbulent_prandtl=.3, filter_width=mode)
            state = model.initial()
            state = model.soa(state*(1+.02*torch.sin(model.s.coords[0]))[..., None])
            native = model.make_stepper('native-p3', block=(2,2,2))
            torch.testing.assert_close(native.plan(state), model.rhs(state), rtol=1e-10, atol=3e-12)
            torch.testing.assert_close(native.plan.viscous(state, model.convection(state)), model.rhs(state), rtol=1e-10, atol=3e-12)
            gradient = model.viscous_operator.gradient(model.primitive(state))
            mu_t = model.sgs.viscosity(state[..., 0], gradient, model.delta)
            expected_ratio = max(float((1+mu_t/model.mu).max()),
                                 float((1+mu_t/model.mu*model.eq.prandtl/.3).max()))
            self.assertGreater(expected_ratio, 10.)
            self.assertAlmostEqual(native.plan.viscous.diffusion_ratio(state), expected_ratio, places=10)
            valid, dt = native.scan(state, .3)
            self.assertTrue(valid)
            self.assertAlmostEqual(dt, model.timestep(state, .3), places=14)
            actual, valid, next_dt = native.advance_finish(state, 0., dt, .3)
            expected = model.make_stepper('torch')(state, 0., dt)
            self.assertTrue(valid)
            torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-12)
            self.assertAlmostEqual(next_dt, model.timestep(actual, .3), places=14)

    def test_half_dof_width_and_scaling(self):
        for p in (0,2,3,4):
            model = TaylorGreenVortexLES((4,)*3, p=p, filter_width='half-dof')
            self.assertEqual(model.sgs_filter_scale, .5)
            self.assertAlmostEqual(model.delta, (2*math.pi/4)/(2*(p+1)), places=14)
        base = TaylorGreenVortexLES((4,)*3)
        half = TaylorGreenVortexLES((4,)*3, filter_width='half-dof')
        state = base.initial()
        gradient = base.viscous_operator.gradient(base.primitive(state))
        old = base.sgs.viscosity(state[..., 0], gradient, base.delta)
        new = half.sgs.viscosity(state[..., 0], gradient, half.delta)
        torch.testing.assert_close(new, .25*old, rtol=0, atol=0)
        self.assertEqual(half.sgs.cv, .07)
        plan = half.make_stepper('native-p3').plan
        self.assertEqual(plan.metadata['sgs_filter_width'], 'half-dof')
        self.assertEqual(plan.metadata['sgs_filter_scale'], .5)

    def test_invalid_center_node_and_restart(self):
        for p in (0,2,4):
            with self.assertRaisesRegex(ValueError, 'zero for even p'):
                TaylorGreenVortexLES((2,)*3, p=p, filter_width='gauss-center')
        with self.assertRaises(ValueError):
            TaylorGreenVortexLES((2,)*3, filter_width='unknown')
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'center'
            options = dict(cells=(2,)*3, final_time=.002, max_dt=.001, filter_width='gauss-center',
                           snapshot_times=(), plot=False, fields=False)
            run(output, max_steps=1, **options)
            actual, _ = run(output, resume=True, **options)
            expected, _ = run(Path(temporary)/'direct', **options)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            config = json.loads((output/'run.json').read_text())['config']
            self.assertEqual(config['model_options']['filter_width'], 'gauss-center')
            for mode in ('dof', 'half-dof'):
                options['filter_width'] = mode
                with self.assertRaises(ValueError):
                    run(output, resume=True, **options)
