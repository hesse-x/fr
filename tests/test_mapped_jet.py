import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from case.round_jet_les.generate_mesh import build_mesh, NOZZLE_RADIUS
from case.round_jet_les.generate_tanh_mesh import write_mesh
from case.round_jet_les.mapped import MappedRoundJetLES
from case.round_jet_les.solve import RoundJetLES
from fr.hex_geometry import HexGeometry
from fr.native.mapped_jet import MappedJetStepper


class MappedJetTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    @staticmethod
    def geometry(edges):
        points, cells, boundaries = {}, [], {}
        sizes = tuple(len(edge)-1 for edge in edges.edges)
        for cell in itertools.product(*(range(n) for n in sizes)):
            nodes = []
            for location in itertools.product((0., .5, 1.), repeat=3):
                point = tuple(float(edge[i]+a*(edge[i+1]-edge[i]))
                              for i, a, edge in zip(cell, location, edges.edges))
                nodes.append(points.setdefault(point, len(points)))
            lattice = torch.tensor(nodes).reshape(3,3,3)
            cells.append(lattice)
            for axis in range(3):
                for side in range(2):
                    if cell[axis] != side*(sizes[axis]-1):
                        continue
                    face = lattice.select(axis, 2*side)
                    key = tuple(sorted(face[::2,::2].flatten().tolist()))
                    boundaries[key] = (side+1 if axis == 0 else 3,
                                       tuple(sorted(face.flatten().tolist())))
        return HexGeometry(torch.tensor(list(points), dtype=torch.float64),
                           torch.stack(cells), torch.ones(len(cells), dtype=torch.long), boundaries,
                           {(2,1):'inlet', (2,2):'outlet', (2,3):'ambient'})

    def test_affine_mapped_rhs_and_rk_match_cartesian(self):
        edges = build_mesh((2, 2, 2), length=2., half_width=1.,
                           axial_stretch=0., transverse_stretch=0.)
        model = RoundJetLES(edges, outlet='nscbc', modal_filter='none')
        geometry = self.geometry(edges)
        native = MappedJetStepper(
            geometry, [(2,2,2)], eq=model.eq, pressure=model.pressure,
            shear_width=model.shear_width, disturbance=model.disturbance, nozzle_radius=NOZZLE_RADIUS,
            outlet='nscbc', threads=1, group_packet=2, modal_filter=False)
        x, y, z = model.s.coords
        velocity = torch.stack((.2*y+.03*x.sin(), .02*z.cos(), .015*x.sin()), -1)
        state = model.soa(model.eq.conservative(1.+.01*x.cos(), velocity,
                                                model.pressure+.02*x.sin()*y.cos()))
        mapped = state.reshape(native.shape)
        t, dt = .54, .001
        out = torch.empty_like(mapped)
        ok, _ = native.scan(mapped, 1.6, t=t)
        self.assertTrue(ok)
        torch.testing.assert_close(native._rhs_buffer.reshape(state.shape), model.rhs(state, t),
                                   atol=3e-11, rtol=2e-10)
        _, ok, _ = native.advance_finish(mapped, t, dt, 1.6, out=out)
        self.assertTrue(ok)
        torch.testing.assert_close(out.reshape(state.shape), model.make_stepper('torch')(state, t, dt),
                                   atol=3e-11, rtol=2e-10)

    def test_square_core_and_curved_ring_match_the_general_operator(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/'mesh.msh'
            write_mesh(path, np.array([0., .21, .45]))
            model = MappedRoundJetLES(path, reynolds=6e5, mach=.5, shear_width=.12,
                                     disturbance=0., modal_filter='none', outlet='nscbc-3d-open')
        mixed = model.make_stepper('native-hex-p3', group_packet=2)
        mapped = MappedJetStepper(
            model.mesh, model.block_shapes, eq=model.eq, pressure=model.pressure,
            shear_width=model.shear_width, disturbance=0., nozzle_radius=NOZZLE_RADIUS,
            outlet=model.outlet, threads=1, group_packet=2, modal_filter=False)
        self.assertEqual(mixed.cartesian_cells, 72)
        self.assertTrue(np.array_equal(mixed.order, mapped.order))
        x, y, z = model.mesh.volume_metrics(cells=mixed.order.tolist()).coordinates.unbind(-1)
        velocity = torch.stack((.1*y.cos(), .04*(x+z).sin(), .03*y.sin()), dim=-1)
        state = model.soa(model.eq.conservative(1.+.01*x.cos(), velocity,
                                                model.pressure+.02*(x+y+z).sin()))
        residuals, advanced, steps = [], [], []
        for solver in (mapped, mixed):
            valid, dt = solver.scan(state, 1.6, t=0.)
            self.assertTrue(valid)
            residuals.append(solver._rhs_buffer.clone())
            out = torch.empty_like(state)
            _, valid, next_dt = solver.advance_finish(state, 0., dt, 1.6, out=out)
            self.assertTrue(valid)
            steps.append((dt, next_dt))
            advanced.append(out)
        torch.testing.assert_close(*residuals, atol=2e-9, rtol=1e-9)
        torch.testing.assert_close(*advanced, atol=2e-10, rtol=1e-10)
        np.testing.assert_allclose(*steps, atol=1e-13, rtol=1e-11)

    def test_filter_step_limit_requests_retry_and_leaves_input_unchanged(self):
        grid = build_mesh((2, 2, 2), length=2., half_width=1., axial_stretch=0., transverse_stretch=0.)
        model = RoundJetLES(grid, disturbance=0.)
        native = MappedJetStepper(
            self.geometry(grid), [(2,2,2)], eq=model.eq, pressure=model.pressure,
            shear_width=model.shear_width, disturbance=0., nozzle_radius=NOZZLE_RADIUS,
            outlet='nscbc-3d-open', threads=1, group_packet=2, modal_filter=False)
        state = model.soa(model.initial()).reshape(native.shape)
        saved = state.clone()
        target = torch.empty_like(state)
        valid, dt = native.scan(state, 1.6, t=0.)
        self.assertTrue(valid)

        evaluate = native._evaluate
        scanned_stages = []

        def evaluate_with_viscous_spike(source, time, *, scan_stage=False, **kwargs):
            valid = evaluate(source, time, scan_stage=scan_stage, **kwargs)
            if scan_stage:
                scanned_stages.append(kwargs['stage'])
                native.scan_values[1] = 1.6/(256*(dt/4))
            return valid

        native._evaluate = evaluate_with_viscous_spike
        native._filter = lambda source: (True, True)
        _, valid, retry = native.advance_finish(state, 0., dt, 1.6, out=target)
        self.assertFalse(valid)
        self.assertEqual(scanned_stages, [1])
        self.assertAlmostEqual(retry, .9*dt/4)
        torch.testing.assert_close(state, saved, atol=0., rtol=0.)

        native._filter = lambda source: (True, False)
        scanned_stages.clear()
        _, valid, _ = native.advance_finish(state, 0., dt, 1.6, out=target)
        self.assertTrue(valid)
        self.assertEqual(scanned_stages, [])

    def test_3d_outlet_projects_transverse_and_reversed_characteristics(self):
        grid = build_mesh((2, 2, 2), length=2., half_width=1.,
                          axial_stretch=0., transverse_stretch=0.)
        model = RoundJetLES(grid, modal_filter='none')
        geometry = self.geometry(grid)
        native = MappedJetStepper(
            geometry, [(2,2,2)], eq=model.eq, pressure=model.pressure,
            shear_width=model.shear_width, disturbance=0., nozzle_radius=NOZZLE_RADIUS,
            outlet='nscbc-3d', threads=1, group_packet=2, modal_filter=False)
        x, y, _ = geometry.volume_metrics().coordinates[native.order].unbind(-1)
        faces = native.data[4] == 5
        ids = native.data[16][faces]
        surface = torch.from_numpy(native.data[2][faces][:, 0])
        scale = .5/float(model.s.R @ model.s.gr)

        for axial, density, lateral in ((.25, torch.ones_like(x), .1*y),
                                         (-.25, 1.+.1*x, .1*x)):
            velocity = torch.stack((torch.full_like(x, axial), lateral, torch.zeros_like(x)), -1)
            state = model.soa(model.eq.conservative(density, velocity, model.pressure))
            ok, _ = native.scan(state, 1.6, t=1.)
            self.assertTrue(ok)
            trace = torch.from_numpy(native.plan[9][ids]).transpose(1, 2)
            rho, vel, p = model.eq.primitive(trace)
            flux = axial*trace
            flux[..., 1] += p
            flux[..., 4] += axial*p
            c = (model.eq.gamma*p/rho).sqrt()
            if axial > 0.:
                mass = -scale*(1.-axial/c)*.1*rho/8
                flux[..., 0] += mass
                flux[..., 1] += mass*(axial-c)
                flux[..., 2] += mass*vel[..., 1]
                flux[..., 4] += mass*((trace[..., 4]+p)/rho-c*axial)
            else:
                mass = -scale*axial*.1
                shear = -scale*axial*.1*rho
                flux[..., 0] += mass
                flux[..., 1] += mass*axial
                flux[..., 2] += mass*vel[..., 1]+shear
                flux[..., 4] += mass*.5*vel.square().sum(-1)+shear*vel[..., 1]
            actual = torch.from_numpy(native.data[17][ids])
            viscous = torch.from_numpy(native.plan[10][ids, :5]).transpose(1, 2)
            torch.testing.assert_close(actual+viscous, surface[..., None]*flux,
                                       rtol=2e-11, atol=2e-12)
            self.assertTrue(torch.all(actual[..., 0]*axial > 0))

    def test_open_faces_switch_characteristics_by_outward_normal(self):
        grid = build_mesh((2, 2, 2), length=2., half_width=1.,
                          axial_stretch=0., transverse_stretch=0.)
        model = RoundJetLES(grid, modal_filter='none')
        geometry = self.geometry(grid)
        native = MappedJetStepper(
            geometry, [(2,2,2)], eq=model.eq, pressure=model.pressure,
            shear_width=model.shear_width, disturbance=0., nozzle_radius=NOZZLE_RADIUS,
            outlet='nscbc-3d-open', threads=1, group_packet=2, modal_filter=False)
        x, y, _ = geometry.volume_metrics().coordinates[native.order].unbind(-1)
        scale = .5/float(model.s.R @ model.s.gr)
        for transverse in (False, True):
            density = torch.ones_like(x) if transverse else 1.+.1*y
            velocity = torch.stack((.1*(x if transverse else y), torch.full_like(y, .25),
                                    torch.zeros_like(y)), -1)
            state = model.soa(model.eq.conservative(density, velocity, model.pressure))
            ok, _ = native.scan(state, 1.6, t=1.)
            self.assertTrue(ok)
            for face in (2,3):
                faces = native.data[4][:,face] == 3
                ids = native.data[16][faces,face]
                surface = torch.from_numpy(native.data[2][faces,face]).norm(dim=1)
                sign = -1. if face == 2 else 1.
                normal_speed = sign*.25
                trace = torch.from_numpy(native.plan[9][ids]).transpose(1,2)
                rho, vel, p = model.eq.primitive(trace)
                flux = normal_speed*trace
                flux[..., 2] += sign*p
                flux[..., 4] += normal_speed*p
                if transverse:
                    c = (model.eq.gamma*p/rho).sqrt()
                    acoustic = -scale*(1.-.25/c)*.1*rho/8
                    flux[..., 0] += acoustic
                    flux[..., 1] += acoustic*vel[..., 0]
                    flux[..., 2] += acoustic*(.25-sign*c)
                    flux[..., 4] += acoustic*((trace[..., 4]+p)/rho-c*normal_speed)
                elif normal_speed < 0.:
                    contact = -scale*normal_speed*sign*.1
                    shear = -scale*normal_speed*sign*.1*rho
                    flux[..., 0] += contact
                    flux[..., 1] += contact*vel[..., 0]+shear
                    flux[..., 2] += contact*vel[..., 1]
                    flux[..., 4] += contact*.5*vel.square().sum(-1)+shear*vel[..., 0]
                viscous = torch.from_numpy(native.plan[10][ids,:5]).transpose(1,2)
                actual = torch.from_numpy(native.data[17][ids])
                torch.testing.assert_close(actual+viscous, surface[..., None]*flux,
                                           rtol=2e-11, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
