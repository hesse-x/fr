"""Constitutive, curved-gradient and conservative viscous balance checks."""
import tempfile
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from airfoil import Airfoil
from generate_mesh import geometry
from fr.gmsh import generate_mesh


class NavierStokesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory()
        geo=Path(cls.tmp.name)/'mesh.geo'; mesh=geo.with_suffix('.msh')
        geo.write_text(geometry(5,.08,3,5)); generate_mesh(geo,mesh,order=3)
        cls.a=Airfoil(mesh)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_shear_and_heat_flux(self):
        a=self.a
        W=torch.tensor([[2.,0.,1.]])
        grad=torch.zeros(1,3,2); grad[0,0,1]=3.; grad[0,2,1]=4.
        F=a.viscous_flux(W,grad)
        self.assertAlmostEqual(float(F[0,1,1]),3*a.mu)
        self.assertAlmostEqual(float(F[0,2,0]),3*a.mu)
        self.assertAlmostEqual(float(F[0,3,1]),6*a.mu+4*a.mu*3.5/a.prandtl)
        self.assertEqual(float(F[0,0].abs().max()),0.)

    def test_constant_rest_curved_gradient(self):
        a=self.a; old=a.far.clone()
        try:
            a.far[1:3]=0.
            U=a.far.expand(a.s.ndof,4).clone()
            self.assertLess(float(a.gradient(a.primitive_vector(U)).abs().max()),1e-8)
            self.assertLess(float(a.viscous(U)[0].abs().max()),1e-8)
        finally: a.far=old

    def test_viscous_balance_and_adiabatic_wall(self):
        a=self.a; s=a.s
        U=a.far.expand(s.ndof,4).clone()
        x,y=s.coords
        U[:,1]+=.001*torch.sin(x)*torch.cos(y)
        rhs,flux=a.viscous(U)
        boundary=a.mesh.boundary
        expected=(flux[boundary]*s.face_jacobian[boundary,:,None]*s.fw[None,:,None]).sum((0,1))
        torch.testing.assert_close(s.integral(rhs),expected,atol=1e-9,rtol=1e-9)
        self.assertEqual(float(flux[a.wall_ids,:,3].abs().max()),0.)
        self.assertEqual(float(rhs[:,0].abs().max()),0.)


    def test_wall_load_decomposition(self):
        a=self.a; U=a.far.expand(a.s.ndof,4).clone()
        loads,wall=a.wall_data(U)
        cl,cd,_=a.loads(U)
        self.assertAlmostEqual(loads['CL'],cl)
        self.assertAlmostEqual(loads['CD_total'],cd)
        self.assertAlmostEqual(loads['CD_total'],loads['CD_pressure']+loads['CD_viscous'])
        self.assertTrue(torch.isfinite(wall).all())



if __name__=='__main__': unittest.main()
