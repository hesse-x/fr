"""Round-jet inputs and output probes for the extruded cylindrical mesh."""
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fr import HexGeometry, NavierStokes
from fr.native.mapped_jet import MappedJetStepper
from fr.operators import operators

from .generate_mesh import NOZZLE_RADIUS


def _quadratic(x):
    return np.stack((.5*x*(x-1), 1-x*x, .5*x*(x+1)), axis=-1)


def _quadratic_slope(x):
    return np.stack((x-.5, -2*x, x+.5), axis=-1)


class ExtrudedProbes:
    def __init__(self, edges, section, order, nodes):
        self.edges, self.section, self.order = edges, section, order
        self.nodes = nodes.numpy()

    def __call__(self, xyz):
        coords = np.column_stack([v.numpy() for v in xyz])
        radial, inverse = np.unique(coords[:, 1:], axis=0, return_inverse=True)
        cross_cells = np.empty(len(radial), dtype=np.int64)
        local = np.empty((len(radial), 2))
        centers = self.section[:, 1, 1, :]
        for start in range(0, len(radial), 256):
            stop = min(start+256, len(radial))
            target = radial[start:stop]
            dist = ((target[:, None, :]-centers[None, :, :])**2).sum(-1)
            candidates = np.argpartition(dist, 15, axis=1)[:, :16]
            candidate_geometry = self.section[candidates]
            reference = np.zeros((*candidates.shape, 2))
            for _ in range(12):
                u, v = np.moveaxis(reference, -1, 0)
                a, b = _quadratic(u), _quadratic(v)
                position = np.einsum('eji,ejk,ejikc->ejc', a, b, candidate_geometry)
                du = np.einsum('eji,ejk,ejikc->ejc', _quadratic_slope(u), b, candidate_geometry)
                dv = np.einsum('eji,ejk,ejikc->ejc', a, _quadratic_slope(v), candidate_geometry)
                err = target[:, None, :]-position
                det = du[..., 0]*dv[..., 1]-du[..., 1]*dv[..., 0]
                usable = np.abs(det) > 1e-14
                denominator = np.where(usable, det, 1.)
                step = np.stack((err[..., 0]*dv[..., 1]-err[..., 1]*dv[..., 0],
                                 du[..., 0]*err[..., 1]-du[..., 1]*err[..., 0]), -1)
                reference = np.clip(reference+np.where(usable[..., None],
                                                       step/denominator[..., None], 0.), -2., 2.)
            u, v = np.moveaxis(reference, -1, 0)
            point = np.einsum('eji,ejk,ejikc->ejc', _quadratic(u), _quadratic(v), candidate_geometry)
            valid = ((np.abs(reference) <= 1.+1e-9).all(-1)
                     & (np.abs(point-target[:, None, :]).max(-1) < 1e-9))
            if not valid.any(1).all():
                raise ValueError('radial probe outside the cylindrical mesh')
            choice = np.where(valid, candidates, len(centers)).argmin(1)
            row = np.arange(stop-start)
            cross_cells[start:stop] = candidates[row, choice]
            local[start:stop] = reference[row, choice]
        x = coords[:,0]
        layer = np.searchsorted(self.edges, x, side='right').clip(1, len(self.edges)-1)-1
        s = 2*(x-self.edges[layer])/(self.edges[layer+1]-self.edges[layer])-1
        locations = np.column_stack((local[inverse], s))
        index = torch.from_numpy(self.order[cross_cells[inverse]*(len(self.edges)-1)+layer])
        z = self.nodes
        denominator = np.prod(np.where(np.eye(len(z), dtype=bool), 1., z[:, None]-z), axis=-1)
        basis = []
        for axis in locations.T:
            numerator = np.prod(np.where(np.eye(len(z), dtype=bool), 1.,
                                         axis[:, None, None]-z), axis=-1)
            basis.append(torch.from_numpy(numerator/denominator))

        def evaluate(field):
            return torch.einsum('mi,mj,mk,mijk...->m...', *basis, field[index])

        return evaluate


class MappedRoundJetLES:
    def __init__(self, path, *, reynolds, mach, shear_width, disturbance, modal_filter, outlet,
                 initial_flow='rest'):
        if (not all(math.isfinite(v) and v > 0 for v in (reynolds, mach, shear_width))
                or not math.isfinite(disturbance) or disturbance < 0):
            raise ValueError('invalid jet Reynolds number, Mach number or inlet profile')
        path = Path(path)
        self.mesh = HexGeometry.read(path)
        with path.open('rb') as stream:
            self.mesh_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.mesh_path = str(path.resolve())
        self.eq = NavierStokes(3, mu=1/reynolds)
        self.mu = self.eq.mu
        self.mach, self.shear_width, self.disturbance = mach, shear_width, disturbance
        self.pressure = 1/(self.eq.gamma*mach**2)
        self.interface_flux = 'rusanov'
        self.modal_filter_name = modal_filter
        self.outlet = outlet
        self.initial_flow = initial_flow
        surface_tags = {name: tag for (dim, tag), name in self.mesh.physical_names.items() if dim == 2}
        cross_cells = int((self.mesh.boundary_tags == surface_tags.get('outlet', -1)).sum())
        side_faces = int((self.mesh.boundary_tags == surface_tags.get('ambient', -1)).sum())
        if not cross_cells or not side_faces:
            raise ValueError('cylinder mesh requires outlet and ambient faces')
        layers, extra_cells = divmod(self.mesh.nelements, cross_cells)
        if not layers or extra_cells:
            raise ValueError('outlet face count must divide the total element count')
        circumferential, extra_faces = divmod(side_faces, layers)
        core_cells, extra_sectors = divmod(circumferential, 4)
        if extra_faces or extra_sectors or not core_cells:
            raise ValueError('expected the five-block round-jet cylinder from its mesh generator')
        radial_cells, extra_ring_cells = divmod(cross_cells-core_cells**2, circumferential)
        if extra_ring_cells or radial_cells < 1:
            raise ValueError('cylinder needs a square core and four matching outer blocks')
        cross_blocks = [(core_cells, core_cells), *((core_cells, radial_cells),)*4]
        self.block_shapes = [(*block, layers) for block in cross_blocks]
        ends = self.mesh.points[self.mesh.cells[:, 1, 1, [0, 2]], 0].reshape(-1, layers, 2).numpy()
        self.edges = np.r_[ends[0, 0, 0], ends[0, :, 1]]
        centers = self.mesh.points[self.mesh.cells[:,1,1,1]].reshape(-1,layers,3).numpy()
        if (abs(self.edges[0]) > 1e-10 or np.any(np.diff(self.edges) <= 0)
                or not np.allclose(ends[...,0], self.edges[:-1], rtol=0, atol=1e-10)
                or not np.allclose(ends[...,1], self.edges[1:], rtol=0, atol=1e-10)
                or not np.allclose(centers[...,0], (self.edges[1:]+self.edges[:-1])/2, rtol=0, atol=1e-10)
                or not np.allclose(centers[...,1:], centers[:,:1,1:], rtol=0, atol=1e-8)):
            raise ValueError('cylinder cells must be stored in axial extrusion order')
        self.section = self.mesh.points[self.mesh.cells[::layers,:,:,1]][...,1:].numpy()
        radius = float(self.mesh.points[:,1:].norm(dim=1).max())
        self.s = SimpleNamespace(p=3, z=operators(3,'cpu')[0],
                                 shape=(self.mesh.nelements,4,4,4), cells=(self.mesh.nelements,),
                                 bounds=((0.,self.edges[-1]),(-radius,radius),(-radius,radius)))
        self.sample_radius = radius-.5
        self.stepper = None

    def checkpoint_mesh(self):
        return {'hex_mesh_sha256': self.mesh_digest}

    def field_geometry(self):
        return dict(mesh_msh=self.mesh_path, mesh_sha256=self.mesh_digest,
                    mesh_cell_order=torch.from_numpy(self.stepper.order),
                    blocks=self.block_shapes)

    def make_stepper(self, backend, *, group_packet):
        if backend != 'native-hex-p3':
            raise ValueError('the cylindrical jet requires native-hex-p3')
        self.stepper = MappedJetStepper(
            self.mesh, self.block_shapes, eq=self.eq, pressure=self.pressure,
            shear_width=self.shear_width, disturbance=self.disturbance, nozzle_radius=NOZZLE_RADIUS,
            outlet=self.outlet, threads=torch.get_num_threads(), group_packet=group_packet,
            modal_filter=self.modal_filter_name != 'none', split_geometry=True)
        if self.initial_flow == 'laminar':
            # Ten inlet periods finish the ramp and preserve all forcing phases.
            self.stepper.inlet_time_offset = 10.
        self.probes = ExtrudedProbes(self.edges, self.section, self.stepper.inverse, self.s.z)
        return self.stepper

    def initial(self):
        if self.initial_flow == 'laminar':
            from .laminar import laminar_state
            return laminar_state(self)
        return self.s.z.new_tensor([1.,0.,0.,0.,self.pressure/(self.eq.gamma-1)]).expand(*self.s.shape,5)

    @staticmethod
    def soa(state):
        return state.movedim(-1,0).contiguous().movedim(0,-1)
