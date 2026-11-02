"""Residual stopping and physical wall-flux invariants for the current solver."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from airfoil import Airfoil
from solve import convergence, parser
from fr.equations import Euler


class CaseTests(unittest.TestCase):
    def test_residual_requires_absolute_and_relative(self):
        args=parser().parse_args([])
        history=[[1,0.,0.,0.,1e-8,3.]]
        self.assertTrue(convergence(history,args,.1)['converged'])
        history[-1][4]=1e-4
        self.assertFalse(convergence(history,args,.1)['converged'])
        history[-1][4]=1e-7
        self.assertFalse(convergence(history,args,1e-4)['converged'])

    def test_wall_flux_equals_reflected_rusanov(self):
        a=Airfoil.__new__(Airfoil);a.eq=Euler(2)
        torch.manual_seed(7)
        n=torch.randn(30,2,dtype=torch.float64);n=n/n.norm(dim=-1,keepdim=True)
        U=torch.tensor([1.,.3,.02,1/1.4/.4+.5*(.3**2+.02**2)],dtype=torch.float64).expand(30,4).clone()
        R=a.boundary(None,U,n,2,0.)
        fL=sum(n[:,k,None]*a.eq.flux(U,k) for k in (0,1))
        fR=sum(n[:,k,None]*a.eq.flux(R,k) for k in (0,1))
        speed=torch.maximum(a.eq.normal_speed(U,n),a.eq.normal_speed(R,n))
        expected=.5*(fL+fR)-.5*speed[:,None]*(R-U)
        actual=torch.zeros_like(U);actual[:,1:3]=a.wall_pressure(U,n)[:,None]*n
        torch.testing.assert_close(expected,actual,atol=1e-14,rtol=0.)


if __name__=='__main__':unittest.main()
