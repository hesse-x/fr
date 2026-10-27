"""Cartesian LDG gradient/divergence with complementary alternating traces.

The gradient has layout (*grid, variables, spatial_dimension). This permits
constitutive laws F(U, grad U), including coupled viscous stresses, rather than
restricting the second-order operator to componentwise scalar diffusion.
"""
import torch


class CartesianLDG:
    """Reuse a CartesianFR grid. Nonperiodic boundaries prescribe Dirichlet U.

    boundary(coords, inside, axis, side, t) returns the boundary VALUE (not
    a ghost state). The dual divergence uses the interior flux there. With
    homogeneous data this gives D = -G^* in the Gauss mass inner product.
    Curved meshes and Neumann/mixed viscous wall conditions are not included.
    """
    def __init__(self, grid, boundary=None):
        self.grid = grid
        self.boundary = boundary
        if not all(grid.periodic) and boundary is None:
            raise ValueError('nonperiodic LDG requires a Dirichlet value callback')

    def derivative(self, U, axis, *, dual=False, t=0.):
        s = self.grid
        if U.shape[:-1] != s.shape:
            raise ValueError('state does not match Cartesian grid')
        perm = s._permutation(axis)
        A = U.permute(perm)
        l, r = s._trace(A, s.L), s._trace(A, s.R)
        # u_hat = u_minus; q_hat = q_plus.
        interior = l[1:] if dual else r[:-1]
        if s.periodic[axis]:
            seam = l[0] if dual else r[-1]
            left, right = seam, seam
        elif dual:
            left, right = l[0], r[-1]
        else:
            cl, cr = s.face_coords[axis]
            left = self.boundary(cl, l[0], axis, -1, t)
            right = self.boundary(cr, r[-1], axis, 1, t)
        common = torch.cat((left[None], interior, right[None]), 0)
        # Explicit dtype conversion also permits complex state tensors.
        d = (A.movedim(s.dim, -1) @ s.D.T.to(U.dtype)).movedim(-1, s.dim)
        shape = [1]*A.ndim
        shape[s.dim] = s.p+1
        d = d+s.gl.reshape(shape)*(common[:-1]-l).unsqueeze(s.dim)
        d = d+s.gr.reshape(shape)*(common[1:]-r).unsqueeze(s.dim)
        return (2/s.cell_width(axis))*d.permute([perm.index(i) for i in range(U.ndim)])

    def gradient(self, U, t=0.):
        return torch.stack([self.derivative(U, a, t=t) for a in range(self.grid.dim)], -1)

    def divergence(self, flux, t=0.):
        if flux.shape[-1] != self.grid.dim:
            raise ValueError('flux must end in (variables, dimension)')
        return sum(self.derivative(flux[..., a], a, dual=True, t=t)
                   for a in range(self.grid.dim))

    def diffusion(self, U, t=0., *, constitutive=None):
        """Return div F; constitutive(coords,U,gradient,t) returns flux.

        Defaults to F=grad U. Boundary flux is interior on Dirichlet faces.
        """
        grad = self.gradient(U, t)
        flux = grad if constitutive is None else constitutive(self.grid.coords, U, grad, t)
        return self.divergence(flux, t)
