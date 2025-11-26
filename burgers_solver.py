import mesh
import solver1d
import torch
import utils
from scipy.special import roots_legendre

stop_step = 10000
skip = 100
order = 2
root_nodes, w = roots_legendre(order + 1)


def compute_flux(u):
    return u * u / 2


def burgers_roe(u: torch.Tensor, flux: torch.Tensor):
    u_L = u[:, 0, 0]
    u_R = u[:, 0, 1]
    f_L = flux[:, 0, 0]
    f_R = flux[:, 0, 1]
    return ((f_L + f_R) / 2 - torch.abs(u_R + u_L) * (u_R - u_L) / 4)[:, None]


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


def eigen_u(u: torch.Tensor):
    return u[:, 0, :]


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


mesh_ = mesh.parse_msh("cases/1d.msh")

equation = solver1d.Equation(F=compute_flux,
                             roe=burgers_roe,
                             init=init_field,
                             eigen_u=eigen_u)
boundary = solver1d.Boundary(fn=cycle_boundary)
config = solver1d.Config(order=order, dim_u=1, CFL=0.1)
correction = utils.SmoothCorrectionFlux(config.order)
solver = solver1d.Solver(mesh_=mesh_,
                         equation=equation,
                         boundary=boundary,
                         correction=correction,
                         config=config)

result = torch.empty((stop_step // skip, solver.elem_prop.elem_ids.shape[0],
                      config.dim_u, config.order + 1),
                     dtype=torch.float64)
for i in range(0, stop_step):
    solver.advance()
    integral = torch.sum(solver.field.u * w[None, None, :])
    if (i % skip == 0):
        result[i // skip, :] = solver.field.u
        print(
            f"Step: {i}, t: {solver.t}, max_u: {torch.max(solver.field.u)}, intergeral: {integral}"
        )
