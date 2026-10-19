import math
from pathlib import Path
import shutil
import tempfile
import unittest
import torch
from fr import Mesh2D,MeshFR2D,normal_central,normal_rusanov_flux
from fr.gmsh import generate_mesh
from fr.reference import Basis,quadrature,edge_points
from fr.operators import operators
from case.mesh_advection import run


class CurvedMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def solver(self,kind='mixed',n=3):
        mesh=Mesh2D.rectangle(n,kind=kind,jitter=.2).mapped(
            lambda x,y:(x+.04*torch.sin(math.pi*x)*torch.sin(math.pi*y),y+.02*torch.sin(math.pi*x)*torch.sin(math.pi*y)),3)
        speed=lambda U,n:(n[...,0]+.3*n[...,1]).abs()
        return MeshFR2D(mesh,flux=lambda U,a:(1.,.3)[a]*U,numerical_flux=normal_rusanov_flux(speed),boundary=lambda c,U,n,tag,t:U,speed=speed)

    def test_reference_sbp_and_dg_lifting(self):
        for kind in ('tri','quad'):
            basis=Basis(kind,3)
            q,w=quadrature(kind,7)
            B=basis.evaluate(q)
            mass=B.T@(w[:,None]*B)
            z,fw=operators(6,'cpu')[:2]
            fp,tangents=edge_points(kind,z/2+.5)
            for axis in (0,1):
                derivative=basis.evaluate(basis.nodes,axis)
                boundary=torch.zeros_like(mass)
                for f,tangent in zip(fp,tangents):
                    trace=basis.evaluate(f)
                    normal=tangent[1] if axis==0 else -tangent[0]
                    boundary+=trace.T@((fw*normal/2)[:,None]*trace)
                torch.testing.assert_close(mass@derivative+derivative.T@mass,boundary,atol=2e-11,rtol=2e-11)

    def test_free_stream_and_affine_solution(self):
        for kind in ('tri','quad','mixed'):
            s=self.solver(kind)
            U=torch.ones(s.ndof,1,dtype=torch.float64)
            self.assertAlmostEqual(s.volume,1.,places=10)
            self.assertLess(s.rhs(U).abs().max().item(),1e-8)
            x,y=s.coords
            U=(x+2*y)[...,None]
            torch.testing.assert_close(s.rhs(U),torch.full_like(U,-1.6),atol=1e-8,rtol=1e-8)

    def test_global_flux_balance_on_mixed_mesh(self):
        s=self.solver()
        torch.manual_seed(3)
        U=torch.randn(s.ndof,2,dtype=torch.float64)
        traces=s._states(U)
        n=s.normals
        boundary=(s.mesh.boundary[:,None]*s.face_jacobian*s.fw)
        expected=-(boundary[...,None]*(n[...,0]+.3*n[...,1])[...,None]*traces).sum((0,1))
        torch.testing.assert_close(s.integral(s.rhs(U)),expected,atol=1e-9,rtol=1e-9)

    def test_triangle_convergence(self):
        errors=[run(n=n,kind='tri',time=.05)[2]['L2'] for n in (4,8)]
        self.assertGreater(errors[0]/errors[1],8.)

    def test_shared_equations_normal_fluxes(self):
        from fr import make_equation,make_normal_numerical_flux,NormalFace
        equations=[make_equation('euler',dim=2),make_equation('linearized-euler',dim=2,mean=(.3,-.2)),
                   make_equation('acoustics',dim=2),make_equation('convected-acoustics',mean=(.3,-.2)),
                   make_equation('maxwell'),make_equation('maxwell-tmz')]
        for eq in equations:
            torch.manual_seed(eq.nvars)
            U=torch.randn(5,eq.nvars,dtype=torch.float64)
            if eq.__class__.__name__=='Euler':
                U[...,0]=1.; U[...,-1]=10.
            for axis in (0,1):
                n=torch.zeros(5,2,dtype=torch.float64);n[...,axis]=1.
                torch.testing.assert_close(eq.normal_speed(U,n),eq.speed(U,axis))
                if hasattr(eq,'absolute_normal_jacobian_action'):
                    torch.testing.assert_close(eq.absolute_normal_jacobian_action(U,None,None,n),eq.absolute_jacobian_action(U,None,None,axis))
                    F=eq.flux(U,axis)
                    face=NormalFace(U,F,(),n,torch.zeros(5),torch.zeros(5))
                    torch.testing.assert_close(make_normal_numerical_flux('roe',eq)(face,face,n),F)

    def test_clockwise_and_bad_cells(self):
        points=[[0.,0.],[1.,0.],[0.,1.]]
        mesh=Mesh2D.from_cells(points,[[0,2,1]])
        self.assertTrue(torch.equal(mesh.blocks[0].vertices,torch.tensor([[0,1,2]])))
        with self.assertRaises(ValueError): Mesh2D.from_cells(points,[[0,0,1]])
        with self.assertRaises(ValueError): Mesh2D.from_cells(points,[[0,1,2]]*3)

    @unittest.skipUnless(shutil.which('gmsh'),'Gmsh CLI not installed')
    def test_real_gmsh_22_41_and_curved_cylinder(self):
        from case.mesh_generation import cylinder_quad_geometry
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            geometry=directory/'cylinder.geo'
            geometry.write_text(cylinder_quad_geometry(angular=2,radial=3,interior=1))
            geometries=[]
            for version in ('2.2','4.1'):
                path=generate_mesh(geometry,directory/f'cylinder{version}.msh',version=version)
                mesh=Mesh2D.read(path)
                s=MeshFR2D(mesh,flux=lambda U,a:U,numerical_flux=normal_central,boundary=lambda c,U,n,tag,t:U)
                self.assertEqual(mesh.physical_names[(2,2)],'dielectric')
                self.assertAlmostEqual(s.volume,25.,places=7)
                volume=sum((b['areas']*(b['region']==2)).sum().item() for b in s.blocks)
                self.assertLess(abs(volume-math.pi*.25),1e-3)
                self.assertGreater(s.minimum_jacobian,0.)
                geometries.append(torch.stack(s.coords,-1))
            torch.testing.assert_close(*geometries,atol=1e-12,rtol=1e-12)

    @unittest.skipUnless(shutil.which('gmsh'),'Gmsh CLI not installed')
    def test_gmsh_triangle_and_reject_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            geo=d/'tri.geo'
            geo.write_text('Point(1)={0,0,0,.5}; Point(2)={1,0,0,.5}; Point(3)={0,1,0,.5}; Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,1}; Curve Loop(1)={1,2,3}; Plane Surface(1)={1}; Physical Surface("domain",1)={1}; Physical Curve("wall",2)={1,2,3};\n')
            mesh=Mesh2D.read(generate_mesh(geo,d/'tri.msh'))
            self.assertEqual({b.kind for b in mesh.blocks},{'tri'})
            s=MeshFR2D(mesh,flux=lambda U,a:U,numerical_flux=normal_central,boundary=lambda c,U,n,tag,t:U)
            self.assertAlmostEqual(s.volume,.5,places=10)
            binary=d/'binary.msh'
            binary.write_bytes(b'$MeshFormat\n4.1 1 8\n$EndMeshFormat\n')
            with self.assertRaisesRegex(ValueError,'ASCII'): Mesh2D.read(binary)


if __name__=='__main__': unittest.main()
