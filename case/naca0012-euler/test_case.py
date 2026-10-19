"""Checks for explicit integration, residual equivalence and convergence gating."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from airfoil import Airfoil
from solve import convergence, parser
from fr import Mesh2D


class CaseTests(unittest.TestCase):
    def test_ssprk43_order(self):
        fake=SimpleNamespace(s=SimpleNamespace(blocks=[{'ns':1}]))
        errors=[]
        for n in (16,32):
            U=torch.ones(1,4,dtype=torch.float64)
            for _ in range(n):
                U=Airfoil.ssprk43(fake,U,torch.tensor([[[1/n]]],dtype=torch.float64),lambda X:-X)
            errors.append(float((U-torch.exp(torch.tensor(-1.,dtype=torch.float64))).abs().max()))
        self.assertGreater(errors[0]/errors[1],7.8)

    def test_rk4_order(self):
        fake=SimpleNamespace(s=SimpleNamespace(blocks=[{'ns':1}]))
        errors=[]
        for n in (8,16):
            U=torch.ones(1,4,dtype=torch.float64)
            for _ in range(n):
                U=Airfoil.rk4(fake,U,torch.tensor([[[1/n]]],dtype=torch.float64),lambda X:-X)
            errors.append(float((U-torch.exp(torch.tensor(-1.,dtype=torch.float64))).abs().max()))
        self.assertGreater(errors[0]/errors[1],15.)

    def test_convergence_uses_adjacent_step_lift_only(self):
        args=parser().parse_args(['--stop-criterion','lift'])
        history=[[0,0.,.5,1.,.1,6.],[50,1.,.5,2.,.2,6.]]
        self.assertFalse(convergence(history,args,.1)['converged'])
        self.assertTrue(convergence(history,args,.1,(.5,.5000001))['converged'])
        self.assertFalse(convergence(history,args,.1,(.5,.500001))['converged'])
        self.assertTrue(convergence(history,args,.1,(0.,0.))['converged'])
        self.assertFalse(convergence(history,args,.1,(0.,1e-15))['converged'])
        self.assertFalse(convergence(history,args,.1,(.5,float('nan')))['converged'])
        self.assertFalse(convergence(history,args,.1,(-.5,.5))['converged'])

    def test_residual_stopping_requires_absolute_and_relative(self):
        args=parser().parse_args([])
        history=[[1,0.,0.,0.,1e-8,6.]]
        self.assertTrue(convergence(history,args,.1)['converged'])
        history[-1][4]=1e-4
        self.assertFalse(convergence(history,args,.1,(0.,0.))['converged'])
        history[-1][4]=1e-7
        self.assertFalse(convergence(history,args,1e-4)['converged'])

    def test_optimized_rhs_matches_repository_and_force_sign(self):
        mesh=Path(__file__).parent/'results/mesh/moderate.msh'
        if not mesh.exists(): self.skipTest('generate moderate.msh first')
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        a=Airfoil(mesh,p=3,alpha=3.,quadrature_order=6)
        torch.manual_seed(123)
        U=a.far.expand(a.s.ndof,4).clone()+1e-5*torch.randn(a.s.ndof,4)
        self.assertLess(float((a.rhs(U)-a.s.rhs(U)).abs().max()),1e-8)
        expected_wall_pressure=a.wall_pressure(a.s._states(U)[a.wall_ids],a.s.normals[a.wall_ids])
        self.assertLess(float((a.loads(U)[2]-expected_wall_pressure).abs().max()),1e-12)
        # p=p_inf-q_inf*y gives a positive vertical pressure force equal
        # to the airfoil's enclosed area (divergence theorem).
        x,y=a.s.coords
        U=a.far.expand(a.s.ndof,4).clone()
        U[:,1:3]=0.
        U[:,3]=(1/1.4-.5*a.mach**2*y)/.4
        cl,cd,_=a.loads(U)
        import math
        cy=cl*math.cos(math.radians(a.alpha))+cd*math.sin(math.radians(a.alpha))
        self.assertAlmostEqual(cy,.0817,places=3)

    def test_wall_flux_equals_reflected_rusanov(self):
        from fr.equations import Euler
        a=Airfoil.__new__(Airfoil)
        a.eq=Euler(2)
        torch.manual_seed(7)
        n=torch.randn(30,2,dtype=torch.float64)
        n=n/n.norm(dim=-1,keepdim=True)
        U=torch.tensor([1.,.3,.02,1/1.4/.4+.5*(.3**2+.02**2)],dtype=torch.float64).expand(30,4).clone()
        R=a.boundary(None,U,n,2,0.)
        fL=sum(n[:,k,None]*a.eq.flux(U,k) for k in (0,1))
        fR=sum(n[:,k,None]*a.eq.flux(R,k) for k in (0,1))
        speed=torch.maximum(a.eq.normal_speed(U,n),a.eq.normal_speed(R,n))
        expected=.5*(fL+fR)-.5*speed[:,None]*(R-U)
        actual=torch.zeros_like(U);actual[:,1:3]=a.wall_pressure(U,n)[:,None]*n
        self.assertLess(float((expected-actual).abs().max()),1e-14)

    def test_neighbor_gather_and_broadcast_steps(self):
        mesh=Path(__file__).parent/'results/mesh/moderate.msh'
        if not mesh.exists():self.skipTest('generate moderate.msh first')
        torch.set_num_threads(1)
        a=Airfoil(mesh,p=1,alpha=3.,quadrature_order=3)
        U=a.far.expand(a.s.ndof,4).clone()
        traces=a.s._states(U)+torch.arange(a.mesh.nfaces,dtype=U.dtype)[:,None,None]*1e-6
        other=traces[a.s.neighbor]
        expected=torch.where(a.s.flip[:,None,None],other.flip(1),other)
        self.assertTrue(torch.equal(a.s.neighbor_states(traces),expected))
        self.assertTrue(a.s.flip.any())
        # Independent flat-state reference retains the previous expanded-dt layout.
        from fr.time import METHODS
        for method in ('ssprk43','rk4'):
            for mode in ('global','local'):
                result,dt=a.make_step(method,mode)(U,U.new_tensor(.2))
                expanded=dt.expand(-1,a.s.blocks[0]['ns'],4).reshape_as(U)
                reference=METHODS[method](lambda X,t:a.s.rhs(X),U,0.,expanded)
                self.assertLess(float((result-reference).abs().max()),1e-12)

    def test_step_fullgraph_with_changing_cfl(self):
        mesh=Path(__file__).parent/'results/mesh/moderate.msh'
        if not mesh.exists():self.skipTest('generate moderate.msh first')
        torch.set_num_threads(1)
        a=Airfoil(mesh,p=1,alpha=3.,quadrature_order=3)
        U=a.far.expand(a.s.ndof,4).clone()
        # Count captures without depending on a C++ compiler in the unit suite.
        for method in ('ssprk43','rk4'):
            for mode in ('global','local'):
                graphs=[]
                def backend(graph,inputs):
                    graphs.append(graph)
                    return graph.forward
                eager=a.make_step(method,mode)
                compiled=torch.compile(eager,backend=backend,fullgraph=True)
                cfl=U.new_tensor(.2)
                with torch.no_grad():
                    for value in (.2,.4):
                        cfl.fill_(value)
                        actual,dt=compiled(U,cfl)
                        expected,reference_dt=eager(U,cfl)
                        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                        torch.testing.assert_close(dt,reference_dt)
                self.assertEqual(len(graphs),1)

    def test_diagnostics_conservation_and_probe_positions(self):
        from diagnostics import Diagnostics
        from tempfile import TemporaryDirectory
        mesh=Path(__file__).parent/'results/mesh/downstream40.msh'
        if not mesh.exists():self.skipTest('generate downstream40.msh first')
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        a=Airfoil(mesh,p=1,alpha=0.,quadrature_order=6)
        U=a.far.expand(a.s.ndof,4).clone()
        with TemporaryDirectory() as folder:
            d=Diagnostics(a,folder)
            result=d.sample(U,a.rhs(U),0,0.,a.local_dt(U,1.))
            self.assertLess(abs(result['mass_balance_error']),1e-9)
            self.assertLess(abs(result['energy_balance_error']),1e-9)
            for probe in d.probes:
                actual=torch.tensor(probe['actual'])
                self.assertLess(float((actual-torch.stack(a.s.coords,-1)[probe['dof_index']]).abs().max()),1e-14)
            d.close()


if __name__=='__main__': unittest.main()
