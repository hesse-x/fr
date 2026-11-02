"""A physical line-voltage functional and its diagonal-mass Riesz lift.

The edge is x=0, y=f, z in [-length/2,length/2]. Average incident DG
traces on a shared edge. Enforce integral E.dl, not a nodal field value.
"""
import math

import torch

from fr.plotting import _lagrange


def voltage_edge(case, length):
    if not math.isfinite(length) or length <= 0:
        raise ValueError('voltage edge length must be finite and positive')
    if length/2 >= case.bounds[2][1] or -length/2 <= case.bounds[2][0]:
        raise ValueError('voltage edge must be inside the domain')
    fixed = (0., case.focal_length)
    owners = []
    for axis, value in enumerate(fixed):
        edges = case.edges[axis]
        ids = ((edges[:-1] <= value+1e-12) & (edges[1:] >= value-1e-12)).nonzero().flatten().tolist()
        owners.append(ids)
    ids, coefficients, masses = [], [], []
    n = case.p+1
    for i in owners[0]:
        hx = float(case.widths[0][i])
        lx = _lagrange(case.z, torch.tensor([2*(fixed[0]-float(case.edges[0][i]))/hx-1], dtype=torch.float64))[0]
        for j in owners[1]:
            hy = float(case.widths[1][j])
            ly = _lagrange(case.z, torch.tensor([2*(fixed[1]-float(case.edges[1][j]))/hy-1], dtype=torch.float64))[0]
            for k in range(case.cells[2]):
                lower = max(-length/2, float(case.edges[2][k]))
                upper = min(length/2, float(case.edges[2][k+1]))
                if upper <= lower+1e-14:
                    continue
                if bool(case.solid[i, j, k]):
                    raise ValueError('voltage edge intersects the PEC reflector')
                hz = float(case.widths[2][k])
                physical = (upper+lower)/2+(upper-lower)/2*case.z
                ref = 2*(physical-case.edges[2][k])/hz-1
                lz = (_lagrange(case.z, ref)*case.w[:, None]).sum(0)*(upper-lower)/2
                weights = lx[:, None, None]*ly[None, :, None]*lz[None, None, :]/(len(owners[0])*len(owners[1]))
                element = (i*case.cells[1]+j)*case.cells[2]+k
                ids.append(element*n**3+torch.arange(n**3, dtype=torch.int64))
                coefficients.append(weights.flatten())
                masses.append((case.weights*hx*hy*hz/8).flatten())
    point_ids = torch.cat(ids).contiguous()
    coefficients = torch.cat(coefficients).contiguous()
    masses = torch.cat(masses)
    lift = coefficients/masses
    lift /= torch.dot(coefficients, lift)
    return point_ids, coefficients, lift.contiguous()


def voltage_signal(t, amplitude=1.):
    ramp = math.sin(math.pi*min(t/2, 1)/2)**2
    derivative = math.pi/4*math.sin(math.pi*t/2) if 0 <= t < 2 else 0.
    angle = 2*math.pi*t
    return amplitude*ramp*math.cos(angle), amplitude*(derivative*math.cos(angle)-2*math.pi*ramp*math.sin(angle))
