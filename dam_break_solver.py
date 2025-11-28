import mesh
import solver1d
import torch
import utils
from scipy.special import roots_legendre

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

stop_step = 10000
skip = 100
g = 9.8
config = solver1d.Config(order=0, dim_u=2, CFL=0.01)
root_nodes, w = roots_legendre(config.order + 1)


def compute_flux(u: torch.Tensor):
    """计算浅水波方程通量（基于守恒变量[h, hu]）"""
    h = u[:, 0, :]  # 水位
    hu = u[:, 1, :]  # 动量（守恒变量，原代码此处为流速u）

    flux_h = hu  # F1 = hu（直接使用守恒变量，省去h*u的乘法）
    # 计算F2 = (hu)^2/h + 0.5gh²，处理h=0避免除零
    h_safe = torch.where(h < 1e-6, 1e-6, h)
    flux_vel = (hu**2) / h_safe + 0.5 * g * h**2
    return torch.stack([flux_h, flux_vel], dim=1)


def lax_friedrichs(u: torch.Tensor, flux: torch.Tensor):
    """Lax-Friedrichs格式（适配守恒变量[h, hu]）"""
    # u形状: [界面数, 变量数, 2]，变量为[h, hu]
    h_L = u[:, 0, 0]
    hu_L = u[:, 1, 0]
    h_R = u[:, 0, 1]
    hu_R = u[:, 1, 1]

    # 从守恒变量计算流速
    h_L_safe = torch.where(h_L < 1e-6, 1e-6, h_L)
    h_R_safe = torch.where(h_R < 1e-6, 1e-6, h_R)
    u_L = hu_L / h_L_safe
    u_R = hu_R / h_R_safe

    # 后续逻辑与原代码一致（基于流速计算特征值）
    c_L = torch.sqrt(g * h_L_safe)
    c_R = torch.sqrt(g * h_R_safe)

    lambda_L1 = u_L - c_L
    lambda_L2 = u_L + c_L
    lambda_R1 = u_R - c_R
    lambda_R2 = u_R + c_R

    alpha = torch.max(
        torch.abs(lambda_L1),
        torch.abs(lambda_L2),
    )
    alpha = torch.max(
        alpha,
        torch.abs(lambda_R1),
    )
    alpha = torch.max(alpha, torch.abs(lambda_R2))

    # 守恒变量差分（直接使用动量差分）
    delta_h = h_R - h_L
    delta_hu = hu_R - hu_L
    delta_U = torch.stack([delta_h, delta_hu], dim=1)

    flux_L = flux[:, :, 0]
    flux_R = flux[:, :, 1]
    lax_friedrichs_flux = 0.5 * (flux_L +
                                 flux_R) - 0.5 * alpha.unsqueeze(1) * delta_U
    return lax_friedrichs_flux


def roe(u: torch.Tensor, flux: torch.Tensor):
    """Roe格式（适配守恒变量[h, hu]）"""
    # u形状: [界面数, 变量数, 2]，变量为[h, hu]（原代码为[h, u]）
    h_L = u[:, 0, 0]  # 左侧水位
    hu_L = u[:, 1, 0]  # 左侧动量（原代码为u_L）
    h_R = u[:, 0, 1]  # 右侧水位
    hu_R = u[:, 1, 1]  # 右侧动量（原代码为u_R）

    # 从守恒变量计算流速（处理h=0）
    h_L_safe = torch.where(h_L < 1e-6, 1e-6, h_L)
    h_R_safe = torch.where(h_R < 1e-6, 1e-6, h_R)
    u_L = hu_L / h_L_safe  # 左侧流速
    u_R = hu_R / h_R_safe  # 右侧流速

    # 守恒变量差分（已正确，无需修改）
    delta_h = h_R - h_L
    delta_hu = hu_R - hu_L  # 直接使用动量差分（原代码需计算h_R*u_R - h_L*u_L）

    # Roe平均（基于流速计算，保持不变）
    sqrt_h_L = torch.sqrt(h_L_safe)
    sqrt_h_R = torch.sqrt(h_R_safe)
    h_roe = sqrt_h_L * sqrt_h_R
    u_roe = (sqrt_h_L * u_L + sqrt_h_R * u_R) / (sqrt_h_L + sqrt_h_R)
    c_roe = torch.sqrt(g * h_roe)

    # 特征值与波幅计算（保持不变）
    lambda1 = u_roe - c_roe
    lambda2 = u_roe + c_roe

    denominator = 2 * c_roe + 1e-6
    alpha1 = ((u_roe + c_roe) * delta_h - delta_hu) / denominator
    alpha2 = (-(u_roe - c_roe) * delta_h + delta_hu) / denominator

    r1 = torch.stack([torch.ones_like(h_roe), u_roe - c_roe], dim=0)
    r2 = torch.stack([torch.ones_like(h_roe), u_roe + c_roe], dim=0)

    flux_diff = 0.5 * (torch.abs(lambda1) * alpha1 * r1 +
                       torch.abs(lambda2) * alpha2 * r2)

    flux_L = flux[:, :, 0]
    flux_R = flux[:, :, 1]
    roe_flux = 0.5 * (flux_L + flux_R) - 0.5 * flux_diff.T
    return roe_flux


def init_field(field: mesh.ElementGroup, prop: mesh.ElementGroupProperty,
               mesh_read: mesh.Mesh):
    """初始化溃坝问题初始条件：左侧高水位，右侧低水位（守恒变量形式）"""
    root_nodes, _ = roots_legendre(config.order + 1)
    for idx, elem_id in enumerate(prop.elem_ids):
        [n0, n1] = mesh_read.inner[elem_id].node_ids
        x0 = mesh_read.nodes[n0][0]
        x1 = mesh_read.nodes[n1][0]
        xl = min(x0, x1)
        xr = max(x0, x1)
        k = (xr - xl) / 2
        b = (xl + xr) / 2
        x_phys = k * root_nodes + b  # 物理空间坐标

        # 初始条件：x < 0 时 h=1.0, u=0；x >=0 时 h=0.1, u=0
        h = torch.where(x_phys < 0, 1.0, 0.1)
        u_vel = torch.zeros_like(h)  # 流速（临时变量）
        hu = h * u_vel  # 动量（守恒变量）

        field.u[idx, 0, :] = h  # 水位（守恒变量）
        field.u[idx, 1, :] = hu  # 动量（守恒变量，原代码此处为流速u）


def eigen_u(u: torch.Tensor):
    """计算时间步长（从守恒变量提取流速）"""
    current_h = u[:, 0, :]
    current_hu = u[:, 1, :]  # 动量（守恒变量）
    # 计算流速u = hu/h，处理h=0
    current_h_safe = torch.where(current_h < 1e-6, 1e-6, current_h)
    current_u = current_hu / current_h_safe
    # 保持原CFL计算逻辑
    return torch.abs(current_u) + torch.sqrt(g * current_h)


def reflect_boundary(interface: mesh.InterfaceGroup,
                     prop: mesh.BoundaryGroupProperty):
    """反射边界条件：法向动量反向（hu → -hu），水位保持不变"""
    for b in prop.boundary_to_interface:
        idx, l_or_r_idx = b[0].item(), b[1].item()
        if l_or_r_idx == 0:  # 左边界
            # 水位不变，动量反向（因u反向，hu = h*u → h*(-u) = -hu）
            interface.u[idx, 0, l_or_r_idx] = interface.u[idx, 0,
                                                          1 - l_or_r_idx]
            interface.u[idx, 1,
                        l_or_r_idx] = -interface.u[idx, 1, 1 - l_or_r_idx]
        else:  # 右边界
            interface.u[idx, 0, l_or_r_idx] = interface.u[idx, 0,
                                                          1 - l_or_r_idx]
            interface.u[idx, 1,
                        l_or_r_idx] = -interface.u[idx, 1, 1 - l_or_r_idx]

        # 重新计算边界通量（基于守恒变量）
        h = interface.u[idx, 0, l_or_r_idx]
        hu = interface.u[idx, 1, l_or_r_idx]  # 此处使用动量hu而非流速u
        interface.flux[idx, 0, l_or_r_idx] = hu
        h_safe = torch.where(h < 1e-6, 1e-6, h)
        interface.flux[idx, 1, l_or_r_idx] = (hu**2) / h_safe + 0.5 * g * h**2


mesh_ = mesh.parse_msh("cases/1d.msh")

equation = solver1d.Equation(
    F=compute_flux,
    #                             roe=lax_friedrichs,
    roe=roe,
    init=init_field,
    eigen_u=eigen_u)
boundary = solver1d.Boundary(fn=reflect_boundary)
correction = utils.CorrectionFlux(config.order, c=0.5)
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
    if (i % skip == 0):
        result[i // skip, :] = solver.field.u
        print(
            f"Step: {i}, Time: {solver.t}, Max h: {solver.field.u[:,0,:].max().item()}"
        )


def compute_x(prop: mesh.ElementGroupProperty, mesh_read: mesh.Mesh):
    x = torch.empty([prop.elem_ids.shape[0], (config.order + 1)],
                    dtype=torch.float64)
    root_nodes, _ = roots_legendre(config.order + 1)
    for idx, elem_id in enumerate(prop.elem_ids):
        [n0, n1] = mesh_read.inner[elem_id].node_ids
        x0 = mesh_read.nodes[n0][0]
        x1 = mesh_read.nodes[n1][0]
        xl = min(x0, x1)
        xr = max(x0, x1)
        k = (xr - xl) / 2
        b = (xl + xr) / 2
        x[idx, :] = k * root_nodes + b
    return torch.reshape(x, [prop.elem_ids.shape[0] * (config.order + 1)])


x = compute_x(solver.elem_prop, solver.mesh_)
# 重塑结果用于绘图：水位h直接取第0分量，流速u需从hu/h计算
result_h = torch.reshape(result[:, :, 0, :], [result.shape[0], -1])  # 水位（直接使用）
# 计算流速结果（从守恒变量hu和h推导）
result_hu = torch.reshape(result[:, :, 1, :], [result.shape[0], -1])
result_h_safe = torch.where(result_h < 1e-6, 1e-6, result_h)
result_u = result_hu / result_h_safe  # 流速（绘图用）

# 绘制动画（保持不变）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 3)
ax1.set_xlabel('x')
ax1.set_ylabel('h')
ax1.set_title('Water Height')

ax2.set_xlim(-1, 1)
ax2.set_ylim(-3, 3)
ax2.set_xlabel('x')
ax2.set_ylabel('u')
ax2.set_title('Velocity')

line1, = ax1.plot(x, result_h[0], color='blue', linewidth=2)
line2, = ax2.plot(x, result_u[0], color='red', linewidth=2)


def update(frame):
    line1.set_ydata(result_h[frame])
    line2.set_ydata(result_u[frame])
    return line1, line2


ani = FuncAnimation(fig,
                    update,
                    frames=result.shape[0],
                    interval=200,
                    blit=True,
                    repeat=True)

plt.tight_layout()
plt.show()
