import mesh
import torch
import numpy as np
from math import factorial, sqrt
from scipy.special import legendre, roots_legendre


def interp_interface(value: torch.Tensor, prop: mesh.ElementGroupProperty):
    A = prop.interp_inteface_matrix
    f = torch.matmul(value, A)
    return f


class CorrectionFlux:
    """VCJH修正函数（仅接收外部传入c，无Mach数/场景依赖）"""
    def __init__(self, order: int, c: float):
        """
        :param order: 多项式阶数p（0,1,2,...）
        :param c: 外部传入的修正参数（用户自行计算后传入）
        """
        self.order = order
        self.c = c  # 仅接收外部传入的c，无内部计算逻辑
        self._gl_derivatives = CorrectionFlux.init_gl(order, c)
        self._gr_derivatives = -torch.flip(self._gl_derivatives, dims=[0])

    @staticmethod
    def init_gl(order, c):
        """核心：根据order和外部传入的c，计算g_L的导数"""
        p = order + 1
        # 零阶特判：常数g_L，导数恒为0
        if p == 0:
            return torch.tensor([-0.5], dtype=torch.float32)

        # 1. 计算η_p（仅依赖外部传入的c）
        a_p = factorial(2 * p) / ((2**p) * factorial(p))
        eta_p = (c * (2 * p + 1) * (a_p**2)) / 2
        # 2. 生成Legendre多项式及解析导数
        Lp_minus1 = legendre(p - 1)
        Lp = legendre(p)
        Lp_plus1 = legendre(p + 1)
        C = ((-1)**p / 2) * (Lp - (eta_p * Lp_minus1 + Lp_plus1) / (1 + eta_p))
        print([C(x) for x in [-1, 0, 1]])
        # 4. Gauss-Lobatto配置点（p+1阶）
        nodes, _ = roots_legendre(p)
        # 5. 计算所有配置点的导数
        dC = C.deriv()
        derivatives = [dC(x) for x in nodes]
        return torch.tensor(derivatives, dtype=torch.float32)

    # 保留原接口，完全兼容你的调用逻辑
    def compute_gr_deri(self, df_r):
        return df_r[:, :, None] * self._gr_derivatives[None, None, :]

    def compute_gl_deri(self, df_l):
        return df_l[:, :, None] * self._gl_derivatives[None, None, :]


class SmoothCorrectionFlux:
    @staticmethod
    def init_gl(order):
        if order == 0:
            return torch.tensor([-0.5], dtype=torch.float32)

        root_nodes0, _ = roots_legendre(order + 1)
        root_nodes = np.append(root_nodes0, 1)
        coeffs = np.poly(root_nodes)
        gl = np.poly1d(coeffs)
        gl = gl / gl(-1)
        gl_deri = gl.deriv()
        derivatives = [gl_deri(x) for x in root_nodes0]
        return torch.tensor(derivatives)

    def __init__(self, order: int):
        self._gl_derivatives = SmoothCorrectionFlux.init_gl(order)
        self._gr_derivatives = -torch.flip(self._gl_derivatives, dims=[0])

    def compute_gr_deri(self, df_r):
        return df_r[:, :, None] * self._gr_derivatives[None, None, :]

    def compute_gl_deri(self, df_l):
        return df_l[:, :, None] * self._gl_derivatives[None, None, :]


class ShockCorrectionFlux(CorrectionFlux):
    """含激波/间断场景修正函数（独立实现：解析推导g_L导数）"""
    def __init__(self, order: int, mach_number: float = 1.0):
        p = self.order + 1
        # 1. 计算激波场景的理论c值（能量稳定下界+马赫数修正）
        c_min = 2 / ((2 * p + 1) * (p + 1))
        c = c_min * max(1.0, sqrt(self.mach_number))
        super().__init__(order, c)
