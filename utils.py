import mesh
import torch
import numpy as np
from scipy.special import roots_legendre


def interp_interface(value: torch.Tensor, prop: mesh.ElementGroupProperty):
    A = prop.interp_inteface_matrix
    f = torch.matmul(value, A)
    return f


class SmoothCorrectionFlux:
    @staticmethod
    def init_g(order):
        root_nodes, _ = roots_legendre(order + 1)
        root_nodes = np.append(root_nodes, 1)
        k = 1 / np.prod(-1 - root_nodes)
        n = len(root_nodes)
        diff = root_nodes[:, np.newaxis] - root_nodes
        np.fill_diagonal(diff, 1)
        derivatives = k * np.prod(diff, axis=1)
        return torch.tensor(derivatives[:-1])

    def __init__(self, order: int):
        self._gl_derivatives = SmoothCorrectionFlux.init_g(order)

    def compute_gr_deri(self, df_r):
        return -df_r[:, :, None] * self._gl_derivatives[None, None, :]

    def compute_gl_deri(self, df_l):
        return df_l[:, :, None] * self._gl_derivatives[None, None, :]
