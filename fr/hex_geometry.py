"""Conforming quadratic hexahedra and their tensor-product FR geometry."""
from dataclasses import dataclass

import torch

from .gmsh import read_ascii_mesh
from .operators import operators


# Complete Gmsh type-12 nodes in (-1, 0, 1)^3 reference coordinates.
_GMSH_NODES = (
    (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
    (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1),
    (0,-1,-1), (-1,0,-1), (-1,-1,0), (1,0,-1), (1,-1,0), (0,1,-1),
    (1,1,0), (-1,1,0), (0,-1,1), (-1,0,1), (1,0,1), (0,1,1),
    (0,0,-1), (0,-1,0), (-1,0,0), (1,0,0), (0,1,0), (0,0,1), (0,0,0),
)
_TENSOR_ORDER = sorted(range(27), key=_GMSH_NODES.__getitem__)


def _face_permutations(n):
    grid = torch.arange(n*n).reshape(n, n)
    # Operations map the neighbor's flattened face order into the local order.
    return torch.stack([plane.flip(dims).flatten()
                        for plane in (grid, grid.T)
                        for dims in ((), (0,), (1,), (0, 1))])


def _faces(lattice):
    return torch.stack([lattice.select(axis+1, side).flatten(1)
                        for axis in range(3) for side in (0, 2)], dim=1)


@dataclass(frozen=True)
class HexMetrics:
    coordinates: torch.Tensor
    jacobian: torch.Tensor
    contravariant: torch.Tensor


class HexGeometry:
    """Q2 geometry at Gauss solution points with oriented, conforming faces.

    Volume coordinates use [cell, i, j, k, physical axis]. Contravariant
    vectors are the rows of J d(reference)/d(physical). A face is numbered
    2*reference_axis + side, with side 0 at -1 and side 1 at +1.
    """

    @classmethod
    def read(cls, path):
        nodes, elements, names = read_ascii_mesh(path)
        indices = {tag: i for i, tag in enumerate(nodes)}
        cells, regions, boundaries = [], [], {}
        for kind, tags, physical in elements:
            if kind in (12, 10):
                if len(tags) != (27 if kind == 12 else 9):
                    raise ValueError('incorrect quadratic element node count')
                local = [indices[tag] for tag in tags]
                if kind == 12:
                    cells.append([local[i] for i in _TENSOR_ORDER])
                    regions.append(physical)
                else:
                    key = tuple(sorted(local[:4]))
                    if key in boundaries:
                        raise ValueError('duplicate surface element')
                    boundaries[key] = (physical, tuple(sorted(local)))
            elif kind not in (1, 8, 15):
                raise ValueError(f'unsupported Gmsh element type {kind}; use hex27 and quad9')
        if not cells:
            raise ValueError('mesh contains no complete quadratic hexahedra')
        points = torch.tensor(list(nodes.values()), dtype=torch.float64)
        return cls(points, torch.tensor(cells).reshape(-1, 3, 3, 3),
                   torch.tensor(regions), boundaries, names)

    def __init__(self, points, cells, regions, boundaries, physical_names):
        if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all():
            raise ValueError('expected finite three-dimensional node coordinates')
        self.points = points
        self.cells = cells
        self.regions = regions
        self.physical_names = physical_names
        self.nelements = len(cells)
        self.neighbor_cell, self.neighbor_face, self.face_transform, self.boundary_tags = (
            self._connect(boundaries))

    def _connect(self, boundaries):
        owners = {}
        records = _faces(self.cells).reshape(-1, 9).tolist()
        for face, nodes in enumerate(records):
            key = tuple(sorted(nodes[i] for i in (0, 2, 6, 8)))
            owners.setdefault(key, []).append(face)

        total = len(records)
        neighbor = [-1]*total
        orientations = [0]*total
        tags = [0]*total
        permutations = _face_permutations(3).tolist()
        inverse = [permutations.index(sorted(range(9), key=p.__getitem__)) for p in permutations]
        matched = set()
        for key, pair in owners.items():
            if len(pair) == 1:
                (face,) = pair
                if key in boundaries:
                    tag, nodes = boundaries[key]
                    if tuple(sorted(records[face])) != nodes:
                        raise ValueError('surface nodes differ from the hexahedron face')
                    tags[face] = tag
                    matched.add(key)
            elif len(pair) == 2 and key not in boundaries:
                left, right = pair
                local, remote = records[left], records[right]
                orientation = next((i for i, p in enumerate(permutations)
                                    if all(local[j] == remote[p[j]] for j in (0, 2, 6, 8))), None)
                if orientation is None or [remote[j] for j in permutations[orientation]] != local:
                    raise ValueError('nonconforming quadratic face')
                neighbor[left], neighbor[right] = right, left
                orientations[left], orientations[right] = orientation, inverse[orientation]
            else:
                raise ValueError('nonmanifold face or tagged interior surface')
        if matched != boundaries.keys():
            raise ValueError('tagged surface has no matching hexahedron face')

        adjacent = torch.tensor(neighbor).reshape(-1, 6)
        cell = torch.where(adjacent >= 0, adjacent//6, -1)
        face = torch.where(adjacent >= 0, adjacent % 6, -1)
        return cell, face, torch.tensor(orientations).reshape(-1, 6), torch.tensor(tags).reshape(-1, 6)

    @staticmethod
    def face_permutations(p):
        """Rows index flattened 2-D Gauss points for each face transform."""
        return _face_permutations(p+1)

    def _evaluate(self, cells, nodes):
        points = self.points[self.cells[cells]]
        value = [torch.stack((.5*x*(x-1), 1-x.square(), .5*x*(x+1)), -1) for x in nodes]
        slope = [torch.stack((x-.5, -2*x, x+.5), -1) for x in nodes]

        def interpolate(basis):
            return torch.einsum('ia,jb,kc,eabcd->eijkd', *basis, points)

        tangents = [interpolate([slope[i] if i == axis else value[i] for i in range(3)])
                    for axis in range(3)]
        cofactors = torch.stack((torch.linalg.cross(tangents[1], tangents[2]),
                                 torch.linalg.cross(tangents[2], tangents[0]),
                                 torch.linalg.cross(tangents[0], tangents[1])), -2)
        jacobian = (tangents[0]*cofactors[..., 0, :]).sum(-1)
        if not torch.isfinite(jacobian).all() or (jacobian <= 0).any():
            raise ValueError('hexahedron mapping must have a positive finite Jacobian')
        return HexMetrics(interpolate(value), jacobian, cofactors)

    def volume_metrics(self, p=3, cells=slice(None)):
        """Evaluate volume geometry at the p+1 Gauss points per direction."""
        z = operators(p, 'cpu')[0]
        return self._evaluate(cells, (z, z, z))

    def face_geometry(self, face, p=3, cells=slice(None)):
        """Evaluate the geometry and its mapping derivatives at Gauss face points."""
        if not 0 <= face < 6:
            raise ValueError('hex face must be in [0, 5]')
        z = operators(p, 'cpu')[0]
        axis, side = divmod(face, 2)
        sign = 2*side-1
        nodes = [z]*3
        nodes[axis] = z.new_tensor([sign])
        metrics = self._evaluate(cells, nodes)
        return HexMetrics(metrics.coordinates.squeeze(axis+1), metrics.jacobian.squeeze(axis+1),
                          metrics.contravariant.squeeze(axis+1))

    def face_metrics(self, face, p=3, cells=slice(None)):
        """Return coordinates and outward area vectors at Gauss face points."""
        metrics = self.face_geometry(face, p, cells)
        axis, side = divmod(face, 2)
        return metrics.coordinates, (2*side-1)*metrics.contravariant[..., axis, :]
