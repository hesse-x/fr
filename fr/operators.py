"""Reference-element FR operators."""
import torch


def overintegration(p, n, device='cpu'):
    """Gauss interpolation and discrete L2 projection, exact mass for P_p."""
    z = operators(p, device)[0]
    q, qw = operators(n-1, device)[:2]
    def legendre(x):
        values = [torch.ones_like(x)]
        if p:
            values.append(x)
        for k in range(1, p):
            values.append(((2*k+1)*x*values[-1]-k*values[-2])/(k+1))
        return torch.stack(values, -1)
    V, Vq = legendre(z), legendre(q)
    interpolation = torch.linalg.solve(V.T, Vq.T).T
    inverse_mass = (2*torch.arange(p+1, device=device, dtype=z.dtype)+1)/2
    projection = (V*inverse_mass) @ (Vq.T*qw)
    return q, qw, interpolation, projection

def operators(p, device):
    """Gauss nodes/weights via Golub-Welsch; Radau derivatives via DG lifting."""
    k = torch.arange(1, p+1, dtype=torch.float64, device=device)
    off = k / (4*k*k-1).sqrt()
    jac = torch.diag(off, 1) + torch.diag(off, -1)
    z, V = torch.linalg.eigh(jac)
    w = 2*V[0]**2
    n = p+1
    diff = z[:, None]-z[None, :]
    diff.fill_diagonal_(1)
    bary = 1/diff.prod(1)
    D = (bary[None, :]/bary[:, None])/diff
    D.fill_diagonal_(0)
    D.diagonal().copy_(-D.sum(1))
    def trace(at):
        a = bary/(at-z)
        return a/a.sum()
    L, R = trace(-1), trace(1)
    # g_L'= -l_i(-1)/w_i, g_R'=l_i(1)/w_i at Gauss points.
    return z, w, D, L, R, -L/w, R/w

