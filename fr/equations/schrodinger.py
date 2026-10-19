"""Real/imaginary two-component form of i psi_t = -1/2 Laplacian psi + V psi."""
import torch


class Schrodinger:
    nvars = 2

    def __init__(self, ldg, potential=None):
        self.ldg = ldg
        self.potential = potential

    def hamiltonian(self, U, t=0.):
        V = 0. if self.potential is None else self.potential(self.ldg.grid.coords, t)
        if torch.is_tensor(V):
            V = V[..., None]
        return -.5*self.ldg.diffusion(U, t)+V*U

    def rhs(self, U, t=0.):
        H = self.hamiltonian(U, t)
        return torch.stack((H[..., 1], -H[..., 0]), -1)

    def probability(self, U):
        return self.ldg.grid.integral(U.square().sum(-1, keepdim=True))[0]

    def energy(self, U, t=0.):
        return self.ldg.grid.integral((U*self.hamiltonian(U, t)).sum(-1, keepdim=True))[0]
