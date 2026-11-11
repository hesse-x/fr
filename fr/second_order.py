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
        centered = getattr(self, 'centered', False)
        interior = .5*(l[1:]+r[:-1]) if centered else (l[1:] if dual else r[:-1])
        if s.periodic[axis]:
            seam = .5*(l[0]+r[-1]) if centered else (l[0] if dual else r[-1])
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


class CartesianBR1Penalty(CartesianLDG):
    """Central lifted gradient/divergence with a separate jump penalty.

    The boundary callback prescribes a value, as for CartesianLDG. Penalties
    use the smaller adjacent cell width and positive-coordinate face jumps.
    ``penalty_divergence`` is separate so coupled systems can map primitive
    jumps to their physical flux components (e.g. NS momentum and heat).
    """
    centered = True

    def __init__(self, grid, boundary=None, *, penalty=4.):
        import math
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError('positive finite BR1 penalty required')
        super().__init__(grid, boundary)
        self.penalty = penalty

    def penalty_divergence(self, U, t=0.):
        s = self.grid
        result = torch.zeros_like(U)
        for axis in range(s.dim):
            perm = s._permutation(axis)
            A = U.permute(perm)
            l, r = s._trace(A, s.L), s._trace(A, s.R)
            interior = l[1:]-r[:-1]
            h = s.widths[axis]
            interior_h = torch.minimum(h[:-1], h[1:])
            if s.periodic[axis]:
                left = right = l[0]-r[-1]
                hl = hr = torch.minimum(h[0], h[-1])
            else:
                cl, cr = s.face_coords[axis]
                left = l[0]-self.boundary(cl, l[0], axis, -1, t)
                right = self.boundary(cr, r[-1], axis, 1, t)-r[-1]
                hl, hr = h[0], h[-1]
            jump = torch.cat((left[None], interior, right[None]), 0)
            face_h = torch.cat((hl[None], interior_h, hr[None]))
            common = self.penalty*(s.p+1)**2*jump/face_h.reshape(-1, *([1]*(jump.ndim-1)))
            shape = [1]*A.ndim
            shape[s.dim] = s.p+1
            value = (s.gl.reshape(shape)*common[:-1].unsqueeze(s.dim)
                     +s.gr.reshape(shape)*common[1:].unsqueeze(s.dim))
            result += (2/s.cell_width(axis))*value.permute([perm.index(i) for i in range(U.ndim)])
        return result

    def diffusion(self, U, t=0., *, constitutive=None):
        if constitutive is not None:
            raise ValueError('coupled BR1 laws must supply their own physical jump penalty')
        return super().diffusion(U, t)+self.penalty_divergence(U, t)
