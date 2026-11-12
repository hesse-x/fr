import importlib
import tempfile
import unittest
from pathlib import Path
import torch
from fr import CartesianMesh
from case.taylor_green_vortex.solve import TaylorGreenVortex
from case.taylor_green_vortex_les.solve import TaylorGreenVortexLES


class CartesianMeshTests(unittest.TestCase):
    def test_roundtrip_and_invalid_edges(self):
        with tempfile.TemporaryDirectory() as d:
            mesh = CartesianMesh.rectangle((2, 3, 4), ((0., 1.), (-1., 1.), (0., 2.)), stretch=1.2)
            loaded = CartesianMesh.read(mesh.write(Path(d)/'mesh.json'))
            self.assertEqual(loaded.cells, mesh.cells)
            for a, b in zip(loaded.edges, mesh.edges):
                self.assertTrue(torch.equal(a, b))
            with self.assertRaisesRegex(ValueError, 'configuration'):
                loaded.require((3, 3, 4), mesh.bounds)
        for edges in ([[0., 0.]], [[0., float('nan')]], [[1., 0.]]):
            with self.assertRaises(ValueError):
                CartesianMesh(edges)

    def test_case_coordinates_match_generated_grid(self):
        channel = importlib.import_module('case.channel-dns.channel').Channel
        import math
        cases = [(TaylorGreenVortex, ((-math.pi, math.pi),)*3, 0., {}),
                 (TaylorGreenVortexLES, ((-math.pi, math.pi),)*3, 0., {}),
                 (channel, ((0., 2*math.pi), (-1., 1.), (0., math.pi)), 1.2, {})]
        with tempfile.TemporaryDirectory() as d:
            for cls, bounds, stretch, options in cases:
                path = CartesianMesh.rectangle((2, 3, 2), bounds, stretch=stretch).write(Path(d)/'mesh.json')
                direct = cls((2, 3, 2), p=1, **options)
                loaded = cls((2, 3, 2), p=1, mesh=path, **options)
                for a, b in zip(direct.s.coords, loaded.s.coords):
                    torch.testing.assert_close(a, b, rtol=0., atol=1e-15)
                torch.testing.assert_close(direct.initial(), loaded.initial(), rtol=0., atol=1e-12)

    def test_repository_meshes(self):
        for name, cells in [('channel-dns', (64, 24, 32)), ('taylor_green_vortex', (64,)*3),
                            ('taylor_green_vortex_les', (32,)*3)]:
            self.assertEqual(CartesianMesh.read(Path('case')/name/'mesh/default.json').cells, cells)

    def test_mesh_restart_checks_edge_coordinates(self):
        import math
        from case.taylor_green_vortex.solve import run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mesh = CartesianMesh.rectangle((2,)*3, ((-math.pi, math.pi),)*3)
            path = mesh.write(root/'mesh.json')
            options = dict(mesh=path, p=1, backend='torch', final_time=0., fields=False, plot=False, snapshot_times=())
            run(root/'output', **options)
            run(root/'output', resume=True, **options)
            mesh.edges[0][1] += .01
            mesh.write(path)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                run(root/'output', resume=True, **options)
