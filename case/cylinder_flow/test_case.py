"""Cylinder geometry, optimized NS equivalence and viscous conservation."""
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import json

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from cylinder import module
from native import NativeCylinder
from generate_mesh import geometry
from fr.gmsh import generate_mesh
from structured_mesh import geometry as structured_geometry
from transfer import project


class CylinderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        geo = Path(cls.tmp.name)/'cylinder.geo'
        geo.write_text(geometry(3,.15,2,4))
        mesh = generate_mesh(geo,geo.with_suffix('.msh'),order=3)
        cls.mesh = mesh
        cls.model = NativeCylinder(mesh)
        cls.reference = module.Airfoil(mesh,reynolds=100.)
        geo=Path(cls.tmp.name)/'structured.geo'
        geo.write_text(structured_geometry(radius=3,downstream=4,box=1.5,
            circumferential=4,radial=3,upstream_cells=4,downstream_cells=8,outer_cells=4))
        cls.quad_mesh=generate_mesh(geo,geo.with_suffix('.msh'),order=3)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def state(self):
        a = self.model
        U = a.initial_state()
        x,y = a.s.coords
        rho,vel,pressure = a.eq.primitive(U)
        rho = rho*(1+.01*torch.sin(x)*torch.cos(y))
        vel = vel+torch.stack((.003*torch.sin(y),.002*torch.cos(x)),-1)
        pressure = pressure*(1+.01*torch.cos(x+y))
        return torch.cat((rho[:,None],rho[:,None]*vel,
                          (pressure/.4+.5*rho*vel.square().sum(-1))[:,None]),-1)

    def test_geometry(self):
        s = self.model.s
        self.assertGreater(float(s.minimum_jacobian),0)
        self.assertAlmostEqual(float(s.volume),42-torch.pi/4,places=5)
        xy = s.face_coordinates[self.model.wall_ids]
        torch.testing.assert_close(xy.square().sum(-1),torch.full_like(xy[...,0],.25),rtol=0,atol=2e-5)

    def test_flux_rhs_and_step_match_reference(self):
        a,b = self.model,self.reference
        U = self.state()
        torch.testing.assert_close(a.rhs(U),b.rhs(U),rtol=1e-10,atol=1e-10)
        fa,fb = a.viscous(U)[1],b.viscous(U)[1]
        torch.testing.assert_close(fa,fb,rtol=1e-12,atol=1e-12)
        cfl = U.new_tensor(.2)
        va,dta,ra = a.make_step_with_rhs()(U,cfl,a.rhs(U))
        vb,dtb,rb = b.make_step_with_rhs()(U,cfl,b.rhs(U))
        torch.testing.assert_close(va,vb,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(dta,dtb)
        torch.testing.assert_close(ra,rb,rtol=1e-10,atol=1e-10)
        self.assertTrue(a.valid(va))

    def test_viscous_conservation_and_adiabatic_wall(self):
        a = self.model; s = a.s
        rhs,flux = a.viscous(self.state())
        boundary = a.mesh.boundary
        expected = (flux[boundary]*s.face_jacobian[boundary,:,None]*s.fw[None,:,None]).sum((0,1))
        torch.testing.assert_close(s.integral(rhs),expected,rtol=1e-10,atol=1e-10)
        self.assertEqual(float(flux[a.wall_ids,:,3].abs().max()),0.)

    def test_native_diagnostics_and_local_step(self):
        a,b = self.model,self.reference
        U=self.state(); F=a.rhs(U)
        self.assertAlmostEqual(a.residual_norm(F),float(a.s.norm(F).max()),places=11)
        self.assertEqual(a.valid(U),b.valid(U))
        for mode in ('global','local'):
            va,da,ra=a.make_step_with_rhs(mode)(U,U.new_tensor(.2),F)
            vb,db,rb=b.make_step_with_rhs(mode)(U,U.new_tensor(.2),b.rhs(U))
            torch.testing.assert_close(va,vb,rtol=1e-10,atol=1e-12)
            torch.testing.assert_close(da,db,rtol=1e-12,atol=1e-15)
            torch.testing.assert_close(ra,rb,rtol=1e-9,atol=1e-10)
        for component,value in ((0,-1.),(3,-1.),(1,float('nan'))):
            bad=U.clone(); bad[0,component]=value
            self.assertFalse(a.valid(bad))

    def test_native_orders_and_oblique_freestream(self):
        for p in (1,3):
            a=NativeCylinder(self.mesh,p=p,mach=.2,alpha=13.,reynolds=40.)
            b=module.Airfoil(self.mesh,p=p,mach=.2,alpha=13.,reynolds=40.)
            U=a.initial_state(); saved=U.clone()
            torch.testing.assert_close(a.rhs(U),b.rhs(U),rtol=1e-9,atol=1e-9)
            torch.testing.assert_close(U,saved,rtol=0,atol=0)
        old=torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            U=self.state()
            torch.testing.assert_close(self.model.rhs(U),self.reference.rhs(U),rtol=1e-10,atol=1e-10)
        finally:
            torch.set_num_threads(old)

    def test_cli_order_and_gif_output(self):
        output=Path(self.tmp.name)/'animation'
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('solve.py')),
            '--mesh',str(self.mesh),'--p','1','--quadrature','4','--mach','.2',
            '--time','0.000001','--max-steps','5','--frames','3','--fps','5',
            '--output',str(output)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        metrics=json.loads((output/'metrics.json').read_text())
        self.assertEqual(metrics['p'],1)
        self.assertEqual(metrics['status'],'final_time')
        from PIL import Image
        with Image.open(output/'vorticity.gif') as gif:
            self.assertGreaterEqual(gif.n_frames,2)
        plot=json.loads((output/'vorticity_plot.json').read_text())
        self.assertAlmostEqual(plot['convective_times'][-1],metrics['convective_time'])

    def test_structured_quad_native_operator(self):
        for p in (1,2):
            a=NativeCylinder(self.quad_mesh,p=p,mach=.2,reynolds=100.)
            b=module.Airfoil(self.quad_mesh,p=p,mach=.2,reynolds=100.)
            self.assertEqual(a.mesh.blocks[0].kind,'quad')
            self.assertGreater(float(a.s.minimum_jacobian),0)
            self.assertAlmostEqual(float(a.s.volume),42-torch.pi/4,delta=1e-5)
            U=a.initial_state();x,y=a.s.coords
            U[:,1]+=.002*torch.sin(y);U[:,2]+=.003*torch.cos(x)
            F=a.rhs(U); reference=b.rhs(U)
            torch.testing.assert_close(F,reference,rtol=1e-9,atol=1e-9)
            self.assertAlmostEqual(a.residual_norm(F),float(a.s.norm(F).max()),places=10)
            for mode in ('global','local'):
                actual=a.make_step_with_rhs(mode)(U,U.new_tensor(.2),F)
                expected=b.make_step_with_rhs(mode)(U,U.new_tensor(.2),reference)
                for v,w in zip(actual,expected):
                    torch.testing.assert_close(v,w,rtol=1e-8,atol=1e-9)
            self.assertTrue(a.valid(actual[0]))

    def test_mesh_projection_preserves_constant_state(self):
        target=NativeCylinder(self.quad_mesh)
        source=self.model
        U=source.far.expand(source.s.ndof,4).clone()
        transferred=project(source,target,U)
        torch.testing.assert_close(transferred,target.far.expand_as(transferred),rtol=1e-11,atol=1e-11)
        back=project(target,source,transferred)
        torch.testing.assert_close(back,U,rtol=1e-11,atol=1e-11)


if __name__ == '__main__':
    unittest.main()
