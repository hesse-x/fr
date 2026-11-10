"""Checks for units, operator assembly and the independent exponential driver."""
import unittest
import numpy as np
import torch
from scipy.sparse import kronsum, diags
from scipy.sparse.linalg import expm_multiply
from case.quantum_double_slit.solver import axis, propagate, uniform_field


class PaperDoubleSlitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chebyshev_matches_sparse_exponential(self):
        x,wx,hx=axis(4,-2,.5,3)
        y,wy,hy=axis(3,-1.5,.5,3)
        potential=100.*((x[:,None]>=0)&(abs(y[None,:])<.5))
        z=np.random.default_rng(9).normal(size=potential.shape).astype(complex)
        z/=np.linalg.norm(z)
        matrix=kronsum(hy,hx)+diags(potential.ravel())
        expected=expm_multiply(-.23j*matrix,z.ravel()).reshape(z.shape)
        propagate(z,hx,hy,potential,.23,.1)
        self.assertLess(np.linalg.norm(z-expected),2e-12)

    def test_paper_units_free_mode_energy(self):
        # A wavelength-one mode has dimensionless kinetic energy one.
        errors=[]
        for h in (.5,.25):
            x,w,kinetic=axis(10,0,h,3)
            psi=np.sqrt(w)*np.sin(2*np.pi*x)
            energy=np.vdot(psi,kinetic@psi).real/np.vdot(psi,psi).real
            errors.append(abs(energy-1))
        self.assertLess(errors[1],errors[0]/10)
        self.assertLess(errors[1],2e-4)

    def test_fr_tensor_interpolation_preserves_polynomial(self):
        x,wx,_=axis(4,-2,.5,3)
        y,wy,_=axis(3,-1.5,.5,3)
        psi=x[:,None]**3+1j*y[None,:]**2
        z=psi*np.sqrt(wx[:,None]*wy[None,:])
        xx,yy,out=uniform_field(z,x,y,wx,wy,.5,3,.1)
        self.assertLess(np.max(abs(out-(xx[:,None]**3+1j*yy[None,:]**2))),1e-13)

    def test_nonuniform_operator_and_parallel_exponential(self):
        ex=np.r_[-2.,np.linspace(-1,1,9),2.]
        ey=np.r_[-2.,np.linspace(-1,1,5),2.]
        x,wx,hx=axis(4,-2,1,3,ex)
        y,wy,hy=axis(4,-2,1,3,ey)
        self.assertAlmostEqual(wx.sum(),4)
        self.assertLess(np.max(abs((hx-hx.T).toarray())),1e-12)
        potential=30.*((abs(x[:,None])<.5)&(abs(y[None,:])<.5))
        z=np.random.default_rng(14).normal(size=potential.shape).astype(complex)
        z/=np.linalg.norm(z)
        expected=expm_multiply(-.17j*(kronsum(hy,hx)+diags(potential.ravel())),z.ravel()).reshape(z.shape)
        other=z.copy()
        propagate(z,hx,hy,potential,.17,.1,threads=1)
        propagate(other,hx,hy,potential,.17,.1,threads=2)
        self.assertLess(np.linalg.norm(z-expected),2e-12)
        self.assertLess(np.linalg.norm(z-other),2e-13)
        physical=x[:,None]**3+1j*y[None,:]**2
        state=physical*np.sqrt(wx[:,None]*wy[None,:])
        xx,yy,out=uniform_field(state,x,y,wx,wy,1,3,.1)
        self.assertLess(np.max(abs(out-(xx[:,None]**3+1j*yy[None,:]**2))),1e-13)


if __name__=='__main__': unittest.main()
