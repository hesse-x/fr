"""Portable Cartesian mesh files: explicit cell edges, independent of FR order."""
import json
import math
from pathlib import Path
import torch


class CartesianMesh:
    def __init__(self, edges):
        self.edges = tuple(torch.as_tensor(e, dtype=torch.float64).cpu() for e in edges)
        if len(self.edges) not in (1, 2, 3):
            raise ValueError('one to three edge arrays required')
        for e in self.edges:
            if e.ndim != 1 or e.numel() < 2 or not torch.isfinite(e).all() or not (e[1:] > e[:-1]).all():
                raise ValueError('finite strictly increasing cell edges required')
        self.cells = tuple(e.numel()-1 for e in self.edges)
        self.bounds = tuple((float(e[0]), float(e[-1])) for e in self.edges)

    @classmethod
    def rectangle(cls, cells, bounds, *, stretch=0.):
        if len(cells) != len(bounds) or not math.isfinite(stretch) or stretch < 0:
            raise ValueError('matching cells/bounds and nonnegative finite stretch required')
        edges = []
        for axis, (n, (a, b)) in enumerate(zip(cells, bounds)):
            if not isinstance(n, int) or n < 1 or not math.isfinite(a) or not math.isfinite(b) or b <= a:
                raise ValueError('positive cell counts and increasing finite bounds required')
            eta = torch.linspace(-1., 1., n+1, dtype=torch.float64)
            if axis == 1 and stretch:
                eta = torch.tanh(stretch*eta)/math.tanh(stretch)
            edges.append(a+(eta+1)*(b-a)/2)
        return cls(edges)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(format='fr-cartesian-v1', edges=[e.tolist() for e in self.edges]), indent=2)+'\n')
        return path

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('format') != 'fr-cartesian-v1':
            raise ValueError('unsupported Cartesian mesh format')
        return cls(data['edges'])

    def require(self, cells, bounds):
        if tuple(cells) != self.cells or len(bounds) != len(self.bounds) or any(
                not math.isclose(a, b, rel_tol=0., abs_tol=1e-12)
                for pair, other in zip(bounds, self.bounds) for a, b in zip(pair, other)):
            raise ValueError('mesh cells/bounds differ from case configuration')
        return self.edges
