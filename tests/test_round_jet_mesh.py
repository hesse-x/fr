import unittest
from pathlib import Path

import torch

from case.round_jet_les.generate_mesh import build_mesh
from fr import CartesianMesh


class RoundJetMeshTests(unittest.TestCase):
    def test_default_mesh_refines_the_inlet_and_axis(self):
        mesh = build_mesh()
        self.assertEqual(mesh.cells, (40, 32, 32))
        self.assertEqual(mesh.bounds, ((0., 16.), (-4., 4.), (-4., 4.)))
        x, y, z = mesh.edges
        self.assertTrue((x.diff().diff() > 0).all())
        for edge in (y, z):
            torch.testing.assert_close(edge, -edge.flip(0), rtol=0., atol=0.)
            outer_widths = edge.diff()[len(edge)//2:]
            self.assertTrue((outer_widths.diff() > 0).all())
        stored = CartesianMesh.read(Path('case/round_jet_les/mesh/cartesian.json'))
        for actual, expected in zip(stored.edges, mesh.edges):
            torch.testing.assert_close(actual, expected, rtol=0., atol=1e-14)

    def test_zero_stretch_supports_odd_and_unequal_cell_counts(self):
        mesh = build_mesh((4, 3, 5), length=8., half_width=2.,
                          axial_stretch=0., transverse_stretch=0.)
        for edge, width in zip(mesh.edges, (2., 4/3, .8)):
            torch.testing.assert_close(edge.diff(), torch.full_like(edge[:-1], width))


if __name__ == '__main__':
    unittest.main()
