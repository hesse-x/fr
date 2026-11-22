import itertools
import unittest

import torch

from fr.hex_geometry import HexGeometry


class HexGeometryTests(unittest.TestCase):
    def test_rotated_shared_face_has_matching_points_and_opposite_normals(self):
        points = {}
        cells = []
        for element in range(2):
            nodes = []
            for x, y, z in itertools.product((0., .5, 1.), repeat=3):
                point = (x, y, z) if element == 0 else (1+x, z, 1-y)
                nodes.append(points.setdefault(point, len(points)))
            cells.append(nodes)
        mesh = HexGeometry(torch.tensor(list(points), dtype=torch.float64),
                           torch.tensor(cells).reshape(2, 3, 3, 3),
                           torch.ones(2, dtype=torch.long), {}, {})
        metrics = mesh.volume_metrics(p=3)
        torch.testing.assert_close(metrics.jacobian, torch.full_like(metrics.jacobian, .125))
        self.assertEqual(int(mesh.neighbor_cell[0, 1]), 1)
        self.assertEqual(int(mesh.neighbor_face[0, 1]), 0)
        self.assertNotEqual(int(mesh.face_transform[0, 1]), 0)
        x, area = mesh.face_metrics(1, cells=[0])
        remote_x, remote_area = mesh.face_metrics(0, cells=[1])
        order = mesh.face_permutations(3)[mesh.face_transform[0, 1]]
        torch.testing.assert_close(x.flatten(1, 2), remote_x.flatten(1, 2)[:, order])
        torch.testing.assert_close(area.flatten(1, 2), -remote_area.flatten(1, 2)[:, order])
        torch.testing.assert_close(area[..., 0], torch.full_like(area[..., 0], .25))


if __name__ == '__main__':
    unittest.main()
