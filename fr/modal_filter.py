"""Density-concentration-triggered exponential modal filtering on hexes.

HiFiLES-inspired three-dimensional extension, not its original 2-D routine.
No pressure clipping or positivity-preserving scaling is performed.
Reference: HiFiLES/HiFiLES-solver, src/eles.cpp concentration CPU routine and
src/eles_quads.cpp concentration matrix construction (checked 2026-11-13).
"""
import math
import numpy as np
import torch


class ConcentrationModalFilter:
    def __init__(self,threshold=.1,alpha=36.,order=2):
        if not math.isfinite(threshold) or threshold<0 or not math.isfinite(alpha) or alpha<=0:
            raise ValueError('finite nonnegative sensor threshold and positive filter alpha required')
        if not isinstance(order,int) or order<2 or order%2:
            raise ValueError('positive even exponential filter order >=2 required')
        self.threshold=threshold;self.alpha=alpha;self.order=order

    def configuration(self):
        return dict(model='hifiles_concentration_3d',threshold=self.threshold,alpha=self.alpha,
            order=self.order,sensor='max directional |pi/N sqrt(1-z^2) d(rho)/dz|^3 N^(3/2)',
            mean_preservation='physical J-weighted conservative average')

    def matrices(self,geometry):
        z=geometry.z.numpy();p=geometry.p;n=p+1
        vandermonde=np.polynomial.legendre.legvander(z,p)
        sigma=np.exp(-self.alpha*(np.arange(n)/p)**self.order);sigma[0]=1.
        matrix=(vandermonde*sigma)@np.linalg.inv(vandermonde)
        sensor=math.pi/n*np.sqrt(1-z*z)[:,None]*geometry.D.numpy()
        return np.ascontiguousarray(matrix),np.ascontiguousarray(sensor)

    def apply(self,geometry,state):
        """Independent Torch reference, returning a new state, sensor and mask."""
        matrix,sensor=self.matrices(geometry)
        matrix=torch.as_tensor(matrix,dtype=state.dtype,device=state.device)
        sensor=torch.as_tensor(sensor,dtype=state.dtype,device=state.device)
        rho=state[...,0];value=torch.zeros(len(state),dtype=state.dtype,device=state.device)
        for axis in range(3):
            derivative=(rho.movedim(axis+1,-1)@sensor.T).abs().pow(3)
            value=torch.maximum(value,derivative.flatten(1).amax(1)*(geometry.p+1)**1.5)
        mask=value>self.threshold;out=state.clone()
        if bool(mask.any()):
            original=state[mask];filtered=original
            for axis in range(3):filtered=(filtered.movedim(axis+1,-1)@matrix.T).movedim(-1,axis+1)
            weight=torch.as_tensor(geometry.J*geometry.weights,dtype=state.dtype,device=state.device)
            weight=weight[mask].reshape((-1,)+(geometry.n,)*3+(1,))
            volume=weight.sum((1,2,3),keepdim=True)
            correction=((original-filtered)*weight).sum((1,2,3),keepdim=True)/volume
            out[mask]=filtered+correction
        return out,value,mask


class CartesianModalFilter:
    """Adapt the branch's hex filter to affine Cartesian cells.

    J is constant within each cell, so it cancels in the normalized correction.
    Keep the hex algorithm and use unit J, including on stretched Cartesian grids.
    """
    def __init__(self, grid, threshold=.1, alpha=36., order=2):
        from types import SimpleNamespace
        if grid.dim != 3 or grid.p < 1:
            raise ValueError('modal filtering requires a 3-D grid with p>=1')
        self.grid = grid
        self.closure = ConcentrationModalFilter(threshold, alpha, order)
        n = grid.p+1
        count = math.prod(grid.cells)
        self.geometry = SimpleNamespace(z=grid.z.detach().cpu(), D=grid.D.detach().cpu(),
            p=grid.p, n=n, J=np.ones((count, n**3)),
            weights=grid.weights.detach().cpu().numpy().reshape(-1))

    def configuration(self):
        return self.closure.configuration()

    def apply(self, state):
        cells = self.grid.cells
        shaped = state.reshape((math.prod(cells),)+(self.grid.p+1,)*3+(5,))
        output, sensor, mask = self.closure.apply(self.geometry, shaped)
        return output.reshape(state.shape), sensor.reshape(cells), mask.reshape(cells)
