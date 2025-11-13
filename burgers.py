import mesh
import elem_map
import utils
import numpy as np
import torch
from scipy.special import roots_legendre
from scipy.interpolate import lagrange

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 配置阶数为2
order = 2
# order = 0, u = 1, dx = 2/40, dt = 2/40, cfl = u * dt / dx => 1
# order = 1, u = 1, dx = 2/40, dt = 2/100, cfl = u * dt / dx => 2 / 5 => 0.4
# order = 2, u = 1, dx = 2/40, dt = 2/420, cfl = u * dt / dx => 2 / 21 ≈ 0.1

# dt = 1 / 50
stop_step = 10000
skip = 100
dim_u = 1
CFL = 0.1

root_nodes, w = roots_legendre(order + 1)

correction = utils.CorrectionFlux(order)


def init_field(field: mesh.ElementGroup, prop: mesh.ElementGroupProperty,
               mesh_read: mesh.Mesh):
    for idx, elem_id in enumerate(prop.elem_ids):
        [n0, n1] = mesh_read.inner[elem_id].node_ids
        x0 = mesh_read.nodes[n0][0]
        x1 = mesh_read.nodes[n1][0]
        xl = min(x0, x1)
        xr = max(x0, x1)
        k = (xr - xl) / 2
        b = (xl + xr) / 2
        field.u[idx, :] = torch.exp(-56 * (k * root_nodes + b)**2)


def compute_flux(u):
    return u * u / 2


def cycle_boundary(interface: mesh.InterfaceGroup,
                   prop: mesh.BoundaryGroupProperty):
    b0 = prop.boundary_to_interface[0]
    b1 = prop.boundary_to_interface[1]
    idx0, l_or_r_idx0 = b0
    idx1, l_or_r_idx1 = b1

    interface.u[idx0, 0, l_or_r_idx0] = interface.u[idx1, 0, 1 - l_or_r_idx1]
    interface.flux[idx0, 0,
                   l_or_r_idx0] = compute_flux(interface.u[idx1, 0,
                                                           1 - l_or_r_idx1])
    interface.u[idx1, 0, l_or_r_idx1] = interface.u[idx0, 0, 1 - l_or_r_idx0]
    interface.flux[idx1, 0,
                   l_or_r_idx1] = compute_flux(interface.u[idx0, 0,
                                                           1 - l_or_r_idx0])


def burgers_roe(u: torch.Tensor, flux: torch.Tensor):
    u_L = u[:, 0, 0]
    u_R = u[:, 0, 1]
    f_L = flux[:, 0, 0]
    f_R = flux[:, 0, 1]
    return (f_L + f_R) / 2 - torch.abs(u_R + u_L) * (u_R - u_L) / 4


def compute_dt(u: torch.Tensor, dx: torch.Tensor):
    current_u = u[:, 0, :]
    return CFL * torch.min(dx[:, :, None] / torch.abs(current_u))


mesh_read = mesh.parse_msh("cases/1d.msh")


def compute_dx(prop: mesh.ElementGroupProperty, mesh_read: mesh.Mesh):
    result = torch.empty((prop.elem_ids.shape[0], ), dtype=torch.float64)
    for idx, elem_id in enumerate(prop.elem_ids):
        [n0, n1] = mesh_read.inner[elem_id].node_ids
        x0 = mesh_read.nodes[n0][0]
        x1 = mesh_read.nodes[n1][0]
        result[idx] = abs(x0 - x1)
    return result[:, None]


# print(mesh_read)

elem_prop, boundary_prop = mesh.get_mesh_groups(mesh_read, order)

elem_prop = elem_prop[0]
boundary_prop = boundary_prop[0]
# print(boundary_prop)
dx_elem = compute_dx(elem_prop, mesh_read)
# print(elem_prop)
interface_prop = mesh.get_interface_property(mesh_read)[0]
# print(interface_prop)

fluxs = mesh.InterfaceGroup(interface_prop, order, dim_u)

field = mesh.ElementGroup(elem_prop, order, dim_u)
init_field(field, elem_prop, mesh_read)

result = torch.empty(
    (stop_step // skip, elem_prop.elem_ids.shape[0], dim_u, order + 1),
    dtype=torch.float64)

t = 0
for i in range(0, stop_step):
    dt = compute_dt(field.u, dx_elem)
    # 插值边界的u
    interped_u = utils.interp_interface(field.u, elem_prop)
    elem_map.scatter_value_to_interface(interped_u, fluxs.u, elem_prop)

    # 计算求解点f
    flux_p = compute_flux(field.u)
    interped_flux = utils.interp_interface(flux_p, elem_prop)
    elem_map.scatter_value_to_interface(interped_flux, fluxs.flux, elem_prop)

    # 计算边界的u/f
    cycle_boundary(fluxs, boundary_prop)

    # 重构通量面通量roe_f
    fluxs.restruct_flux.copy_(burgers_roe(fluxs.u, fluxs.flux)[:, None])

    # 计算求解点通量导数
    elem_bound_flux_diff = elem_map.gather_interface_vaue(
        fluxs.restruct_flux[:, :, None] - fluxs.flux, elem_prop, dim_u)
    field.flux_deri = torch.matmul(
        flux_p, elem_prop.deri_matrix) + correction.compute_gl_deri(
            elem_bound_flux_diff[:, :, 0]) + correction.compute_gr_deri(
                elem_bound_flux_diff[:, :, 1])

    # 推进u
    field.u = field.u - field.flux_deri * dt * 2 / dx_elem[:, :, None]

    integral = torch.sum(field.u * w[None, None, :])
    t += dt
    if (i % skip == 0):
        result[i // skip, :] = field.u
        print(
            f"Step: {i}, t: {t}, max_u: {torch.max(field.u)}, intergeral: {integral}"
        )
    # print("i:", i, "\nu:", field.u)
    # print(fluxs)
# print(field)


def compute_x(prop: mesh.ElementGroupProperty, mesh_read: mesh.Mesh):
    x = torch.empty([elem_prop.elem_ids.shape[0], (order + 1)],
                    dtype=torch.float64)
    for idx, elem_id in enumerate(prop.elem_ids):
        [n0, n1] = mesh_read.inner[elem_id].node_ids
        x0 = mesh_read.nodes[n0][0]
        x1 = mesh_read.nodes[n1][0]
        xl = min(x0, x1)
        xr = max(x0, x1)
        k = (xr - xl) / 2
        b = (xl + xr) / 2
        x[idx, :] = k * root_nodes + b
    return torch.reshape(x, [elem_prop.elem_ids.shape[0] * (order + 1)])


x = compute_x(elem_prop, mesh_read)
result = torch.reshape(
    result, [result.shape[0], elem_prop.elem_ids.shape[0] * (order + 1)])
# print(x)
# print(result)

# fig, ax = plt.subplots(figsize=(8, 4))
# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 2)
# ax.set_xlabel('t')
# ax.set_ylabel('u')
# ax.set_title('u-t')
#
# line, = ax.plot(x, result[0], color='blue', linewidth=2)
#
#
# def update(frame):
#     line.set_ydata(result[frame])
#     return line,
#
#
# ani = FuncAnimation(fig,
#                     update,
#                     frames=result.shape[0],
#                     interval=200,
#                     blit=True,
#                     repeat=True)
#
# plt.show()
