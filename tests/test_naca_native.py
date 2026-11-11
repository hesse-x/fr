"""Native NACA operators against independent Torch Euler/BR1 NS physics."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from fr.gmsh import generate_mesh
from tests.compressible_reference import CompressibleReference

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeNacaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.modules = {name: load(f'case/naca0012-{name}/native.py', f'test_native_{name}')
                       for name in ('euler', 'ns')}
        quad = load('case/naca0012-euler/structured_mesh.py', 'test_naca_quad_mesh')
        cls.quad = quad
        cls.meshes = {}
        for kind, text in (
            ('quad', quad.geometry(radius=5., downstream=6., surface_cells=6,
                                   radial=5, downstream_cells=10, outer_cells=6)),):
            geo = cls.directory/f'{kind}.geo'
            mesh = geo.with_suffix('.msh')
            geo.write_text(text)
            generate_mesh(geo, mesh, order=3)
            cls.meshes[kind] = mesh

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pair(self, name, kind='quad', p=2):
        m = self.modules[name]
        kwargs = dict(p=p, mach=.2, alpha=4., quadrature_order=p+2)
        if name == 'ns':
            kwargs.update(reynolds=1000., viscous_step_factor=2.)
        native = m.NativeAirfoil(self.meshes[kind], **kwargs)
        reference = CompressibleReference(m.module.Airfoil(self.meshes[kind], **kwargs))
        torch.manual_seed(721)
        U = native.far.expand(native.s.ndof, 4).clone()
        U += .0001*torch.randn_like(U)
        return native, reference, U

    def test_rhs_validity_and_residual_all_orders_and_cells(self):
        for name in ('euler', 'ns'):
            for kind in self.meshes:
                for p in (1, 2, 3):
                    with self.subTest(physics=name, kind=kind, p=p):
                        a, r, U = self.pair(name, kind, p)
                        actual = a.rhs(U)
                        torch.testing.assert_close(actual, r.rhs(U), atol=2e-10, rtol=3e-11)
                        self.assertEqual(a.valid(U), r.valid(U))
                        self.assertAlmostEqual(a.residual_norm(actual), float(r.s.norm(actual).max()), places=11)
                        bad = U.clone(); bad[0, 0] = -1.
                        self.assertFalse(a.valid(bad))
                        bad = U.clone(); bad[0, 3] = float('nan')
                        self.assertFalse(a.valid(bad))
                        with self.assertRaises(RuntimeError):
                            a.rhs(U.float())

    def test_complete_rk_steps_and_changed_cfl(self):
        for name in ('euler', 'ns'):
            for mode in ('global', 'local'):
                for method in (('ssprk43', 'rk4') if name == 'euler' else ('ssprk43',)):
                    a, r, U = self.pair(name)
                    V = U.clone()
                    if name == 'ns':
                        an, ar = a.make_step_with_rhs(mode), r.make_step_with_rhs(mode)
                        fn, fr = a.rhs(U), r.rhs(V)
                    else:
                        an, ar = a.make_step(method, mode), r.make_step(method, mode)
                    for cfl in (.05, .1, .08):
                        c = U.new_tensor(cfl)
                        if name == 'ns':
                            U, dn, fn = an(U, c, fn)
                            V, dr, fr = ar(V, c, fr)
                            torch.testing.assert_close(fn, fr, atol=2e-10, rtol=3e-10)
                        else:
                            U, dn = an(U, c); V, dr = ar(V, c)
                        torch.testing.assert_close(U, V, atol=2e-12, rtol=2e-12)
                        torch.testing.assert_close(dn, dr, atol=1e-15, rtol=2e-13)

    def test_thread_consistency(self):
        for name in ('euler', 'ns'):
            a, r, U = self.pair(name)
            serial = a.rhs(U)
            torch.set_num_threads(2)
            try:
                torch.testing.assert_close(a.rhs(U), serial, atol=0., rtol=0.)
            finally:
                torch.set_num_threads(1)

    def test_ldg_viscous_native_and_adiabatic_wall(self):
        m = self.modules['ns']
        a = m.NativeAirfoil(self.meshes['quad'], p=2, mach=.2, alpha=7., viscous_scheme='ldg')
        r = CompressibleReference(m.module.Airfoil(self.meshes['quad'], p=2,
            mach=.2, alpha=7., viscous_scheme='ldg'))
        torch.manual_seed(109)
        U = a.far.expand(a.s.ndof, 4).clone()+.0001*torch.randn(a.s.ndof, 4, dtype=torch.float64)
        torch.testing.assert_close(a.rhs(U), r.rhs(U), atol=2e-10, rtol=3e-11)
        residual, faces = a.viscous(U)
        self.assertEqual(float(faces[a.wall_ids, :, -1].abs().max()), 0.)
        exterior = a.mesh.boundary
        total = (faces[exterior]*a.s.face_jacobian[exterior, :, None]*a.s.fw[None, :, None]).sum((0, 1))
        torch.testing.assert_close(a.s.integral(residual), total, atol=2e-10, rtol=1e-10)
        for mode in ('global', 'local'):
            native, reference = a.make_step_with_rhs(mode), r.make_step_with_rhs(mode)
            V, W = U, U
            fn, fr = a.rhs(V), r.rhs(W)
            for cfl in (.05, .08):
                V, dn, fn = native(V, U.new_tensor(cfl), fn)
                W, dr, fr = reference(W, U.new_tensor(cfl), fr)
                torch.testing.assert_close(V, W, atol=3e-12, rtol=3e-12)
                torch.testing.assert_close(dn, dr, atol=1e-15, rtol=3e-13)

    def test_cached_scan_mutation_cfl_and_output_ownership(self):
        for name in ('euler', 'ns'):
            a, r, U = self.pair(name)
            pointers = [t.data_ptr() for t in a._native_workspace]
            self.assertTrue(a.valid(U))
            self.assertIsNotNone(a._cached_scan(U))
            # A shared-storage view mutation must invalidate a cached scan.
            U.view(-1)[0] += .001
            self.assertIsNone(a._cached_scan(U))
            V, dn = a.make_step('ssprk43','local')(U,U.new_tensor(.07))
            R, dr = r.make_step('ssprk43','local')(U,U.new_tensor(.07))
            torch.testing.assert_close(V,R,atol=2e-12,rtol=2e-12)
            saved = V.clone()
            # Independent CFL changes reuse speed/rho, not a cached time step.
            self.assertTrue(a.valid(V))
            W, dn = a.make_step('ssprk43','local')(V,V.new_tensor(.03))
            Q, dr = r.make_step('ssprk43','local')(R,R.new_tensor(.03))
            torch.testing.assert_close(W,Q,atol=2e-12,rtol=2e-12)
            torch.testing.assert_close(dn,dr,atol=1e-15,rtol=2e-13)
            torch.testing.assert_close(V,saved,atol=0.,rtol=0.)
            self.assertEqual(pointers,[t.data_ptr() for t in a._native_workspace])
            with torch.inference_mode():
                Z = U.clone()
                self.assertTrue(a.valid(Z))
                self.assertIsNone(a._cached_scan(Z))
                a.make_step()(Z,Z.new_tensor(.03))

    def test_structured_geometry_and_refinement_counts(self):
        from fr import Mesh2D, MeshFR2D, Euler
        from fr.unstructured import make_normal_numerical_flux
        mesh = Mesh2D.read(self.meshes['quad'])
        self.assertEqual({b.kind for b in mesh.blocks}, {'quad'})
        eq = Euler(2)
        s = MeshFR2D(mesh, p=1, flux=eq.flux,
                    numerical_flux=make_normal_numerical_flux('rusanov', eq),
                    boundary=lambda coords,U,n,tag,t: U)
        # 8 wall patches, 4 semicircle patches, 2 vertical-arm patches,
        # plus 4 downstream patches; every interface is conforming.
        expected = 8*6*5+6*6*6+2*10*(6+6)
        self.assertEqual(len(mesh.blocks[0].geometry), expected)
        self.assertGreater(s.minimum_jacobian, 0.)
        n = s.normals[mesh.boundary]
        w = s.face_jacobian[mesh.boundary]*s.fw
        torch.testing.assert_close((n*w[..., None]).sum((0, 1)), torch.zeros(2,dtype=n.dtype), atol=1e-10, rtol=0.)
        # Independent analytic domain and NACA thickness area.
        foil_area = 1.2*(.2969*2/3-.126/2-.3516/3+.2843/4-.1036/5)
        exact = .5*torch.pi*25+10*(7-.5)-foil_area
        self.assertLess(abs(s.volume-exact), 2e-4)
        for kwargs in (dict(radius=2.),dict(surface_cells=1),dict(cross_half_width=.01)):
            with self.assertRaises(ValueError):
                self.quad.geometry(**kwargs)

    def test_native_cli_and_restart(self):
        for name in ('euler', 'ns'):
            out = self.directory/f'run-{name}'
            args = [sys.executable, str(ROOT/f'case/naca0012-{name}/solve.py'),
                    '--mesh', str(self.meshes['quad']), '--output', str(out),
                    '--threads', '1', '--p', '1', '--quadrature', '3',
                    '--cfl', '.05', '--max-steps', '2']
            if name == 'euler':
                args += ['--sample-every', '1', '--window', '2']
            else:
                args += ['--viscous-scheme', 'ldg']
            for restart in (False, True):
                command = args if not restart else [*args, '--restart', str(out/'solution.pt'), '--max-steps', '4']
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            data = torch.load(out/'solution.pt', weights_only=False)
            self.assertEqual(data['stats']['steps'], 4)
            self.assertIn('native', data['stats']['implementation'])
            if name == 'ns':
                self.assertEqual(data['stats']['viscous_scheme'], 'ldg')
                changed = [*args, '--restart', str(out/'solution.pt'), '--viscous-scheme', 'br1-penalty']
                result = subprocess.run(changed, cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('same viscous scheme', result.stderr)


if __name__ == '__main__':
    unittest.main()
