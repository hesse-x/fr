# PyTorch Flux Reconstruction

基于 PyTorch 的高阶 FR-DG 数值求解框架，支持 Cartesian 网格上的一至三维计算，以及二维曲边三角形、四边形和混合网格。提供 Euler、线化 Euler、声学、Maxwell 和含时薛定谔方程算例，用于验证空间离散、时间推进、边界处理和主要物理量的精度。

求解核心依赖 PyTorch，默认使用 CPU、float64；PNG/GIF 可视化使用 Matplotlib 和 Pillow。CPU 已实测，GPU 尚未实测。

- [支持功能](#支持功能)
- [框架设计](#框架设计)
- [算例](#算例)

## 支持功能

### 功能范围

| 类别 | 当前支持 |
|---|---|
| 一阶空间离散 | 1D/2D/3D Cartesian FR-DG；二维非结构曲边 DG/FR-DG |
| 二阶空间离散 | Cartesian LDG 梯度、散度、Laplacian 和自定义耦合扩散通量 |
| 物理方程 | 非线性 Euler、均匀背景线化 Euler、静止/均匀对流声学、Maxwell、含时薛定谔 |
| 时间推进 | SSPRK(4,3)、经典 RK4、自定义步进；固定或回调时间步 |
| 界面通量 | 中心、Rusanov；具备特征分解的系统支持 Roe/迎风；Cartesian Richtmyer 预测通量 |
| 网格 | Cartesian 张量积；二维共形三角形/四边形/混合网格；ASCII Gmsh MSH 2.2/4.1 |
| 边界与源项 | 周期、给定外部状态、外推、声学刚壁、Maxwell PEC、标签组合、体源和方向分裂 PML；LDG 支持周期/Dirichlet |
| 数据与图像 | 按算例输出 JSON、PT、CSV、SVG、VTK，以及 PNG 云图和 GIF 动画 |
| 验证 | 解析解误差、网格收敛、离散守恒/能量收支、扩大域参考、双缝独立有限差分对照 |

默认多项式次数通常为 `p=3`。光滑问题的实际收敛阶取决于方程、通量、边界和时间误差，不能统一宣称四阶；各算例给出实际结果。

### 安装与运行

从项目根目录运行；已有 `venv` 时直接激活，否则先创建：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-plot.txt

# 查看入口参数并运行一个算例
python -m case.ringleb --help
python -m case.ringleb --n 8 --p 3 --time .1

# 完整回归测试
python -m unittest discover -s tests -v
```

仅做数值计算可安装 `requirements.txt`，运行时使用相应入口的 `--no-plot`。Gmsh 网格生成另外需要 PATH 中的 `gmsh` 可执行文件，读入已有网格无需调用 Gmsh。

`--n` 通常表示每个方向的单元数，`--p` 为多项式次数；不同入口的参数以 `--help` 为准。Cartesian 每单元有 `(p+1)^d` 个解点。部分入口提供 `--threads`、`--device`、`--cfl`；默认的小网格计算通常使用 CPU 单线程。

### 适用范围与待支持功能

- Euler 尚无激波限制器、正性保持器、非线性去混叠或 Roe 通量；非物理密度/压力会报错。
- 二维非结构离散实现 DG/FR-DG 成员，尚无完整 VCJH 参数族、非共形面或三维非结构体网格。
- 二阶 LDG 尚未扩展到曲边非结构网格，也未实现 Neumann/混合条件。完整 NS 方程、黏性壁面和热边界仍需后续实现。
- 固定网格斜向背景流直接复用 PML 属于数值实验，其一般稳定性和长时间完美匹配尚未证明。

## 框架设计

### 模块分工与计算流程

```text
case/                         初值、解析解、物理参数、边界组装、命令行入口
  ↓ 组装
fr/equations/                 物理方程、特征速度与诊断
fr/fluxes.py + boundaries.py  公共界面通量与边界回调
  ↓ 空间离散
fr/core.py                    CartesianFR：1–3 维张量积 FR-DG
fr/second_order.py            CartesianLDG：复用 Cartesian 网格构造二阶算子
fr/unstructured.py            MeshFR2D：二维曲边 DG/FR-DG
  ↓ 半离散右端 rhs(U,t)
fr/time.py                    RK4 / SSPRK(4,3) / 自定义步进
  ↓ 结果与诊断
fr/output.py + plotting.py    数据导出、云图和动画
```

`fr/operators.py` 提供 Gauss 点、插值/微分矩阵和 Radau 修正；`fr/reference.py` 提供二维参考单元基与积分规则；`fr/mesh.py` 管理几何和邻接；`fr/gmsh.py` 提供网格生成工具。算例模块可以安全导入，导入时不会启动计算。

### Cartesian FR-DG 离散

Cartesian 离散对应 Huynh (2007), *A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods* 中的 FR-DG 方法。

- Cartesian 张量积单元，张量积 Legendre–Gauss 解点，重心 Lagrange 插值和微分。
- 论文式 (3.8)：`F = f^D + (f*_L-f^D_L)g_L + (f*_R-f^D_R)g_R`。
- `g_L=(-1)^p(P_p-P_{p+1})/2`，`g_R=(P_p+P_{p+1})/2`。在 Gauss 解点用等价的 Radau 导数 `g'_L=-l_i(-1)/w_i`、`g'_R=l_i(1)/w_i` 实现。
- 面上由插值得到左右**状态**后计算所选公共通量（例如 Rusanov）；修正量中减去的是插值得到的**离散通量**，保证非线性情形下的离散守恒。
- 按论文第 VIII 节逐空间方向分别重构，乘物理尺度 `2/h` 后相加。
- 时间推进与空间离散分离。双曲算例通常默认 SSPRK(4,3)，薛定谔算例使用 RK4；算例不通过精确残差抵消制造参考解。

### Cartesian 回调与时间推进

目前通用空间离散支持 1–3 维 Cartesian 张量积网格，每个方向可设置不同单元数和区域范围，单个方向等距。所有方向使用相同多项式次数。曲边和非结构二维网格使用独立的 `MeshFR2D` 接口，见下文。

#### 用 lambda 配置通量

以下代码可以直接运行，展示一维周期平流 `u_t + (2u)_x = 0`：

```python
import torch
from fr import CartesianFR, integrate

s = CartesianFR(
    cells=(16,), bounds=((0., 1.),), p=3,
    flux=lambda U, axis: 2*U,           # 解点物理通量
    face_flux=lambda U, axis: 2*U,      # 面上物理通量；省略则使用 flux
    flux_pair=lambda Fminus, Fplus: Fminus,  # 正速度的迎风公共通量
    speed=lambda U, axis: torch.full_like(U[..., 0], 2.),
    periodic=True,
)
U0 = torch.sin(2*torch.pi*s.coords[0])[..., None]
U, stats = integrate(
    s.rhs, U0, final_time=.1,
    dt=lambda U, t: s.timestep(U, cfl=.15), method='ssprk43',
)
```

维数由 `cells` 的长度确定，例如二维 `(8,12)`、三维 `(4,6,8)`，并提供对应数量的 `bounds`。状态张量布局统一为 `[*单元维度, *解点维度, 变量]`，标量方程也保留末尾长度为 1 的变量维度；`s.coords` 是物理坐标张量的 tuple，`s.shape` 是去掉变量轴后的形状。数值计算默认使用 float64 解点，用户可直接据此构造初值。

通量接口约定如下：

| 回调 | 参数与返回值 |
|---|---|
| `flux(U, axis)` | 输入守恒状态，返回同形状的正坐标方向物理通量 |
| `face_flux(U, axis)` | 输入插值到面的状态，返回同形状物理通量；默认与 `flux` 相同 |
| `flux_pair(Fminus, Fplus)` | 输入两个同方向物理通量张量，返回同形状公共通量；与 `numerical_flux` 二选一 |
| `numerical_flux(minus, plus, axis)` | 输入两个 `Face`（具有 `.state`、`.flux`），返回公共通量，可访问状态和方向 |
| `boundary(coords, inside, axis, side, t)` | 返回域外状态，与 `inside` 同形状；`side=-1/+1` 表示下/上边界，`coords` 与 `inside` 的空间形状对应 |
| `speed(U, axis)` | 返回去掉变量轴后的非负最大特征速度，用于可选 CFL 估计 |

`minus` 始终为坐标较小的一侧，`plus` 为较大的一侧；即使在域下边界，传入的通量也沿正坐标方向，用户不需要乘外法向符号。`flux_pair` 不接收方向；需要按轴选择迎风侧时使用 `numerical_flux`。

仅有两个物理通量不足以构造一般非线性方程的 Rusanov 耗散，因为耗散还需要 `Uplus-Uminus` 和特征速度。对此可以使用富接口（所有入口同样接受 lambda）：

```python
from fr import CartesianFR, Euler, rusanov_flux

e = Euler(dim=3, gamma=1.4)
s = CartesianFR(
    cells=(4, 6, 8), bounds=((0., 1.),)*3, p=3,
    flux=lambda U, axis: e.flux(U, axis),
    numerical_flux=rusanov_flux(lambda U, axis: e.speed(U, axis)),
    speed=lambda U, axis: e.speed(U, axis),
    periodic=True,
)
```

`rusanov_flux(speed)` 返回的函数等价于：

```python
lambda minus, plus, axis: (
    .5*(minus.flux + plus.flux)
    - .5*torch.maximum(speed(minus.state, axis), speed(plus.state, axis))[..., None]
      *(plus.state - minus.state)
)
```

面通量用面状态重新计算；FR 修正中减去的仍是从单元内插值得到的离散通量。这两者对非线性方程通常不同，保留这种区别才能得到正确的守恒离散。公共通量需要满足一致性（相同两侧状态时还原物理通量）。

非周期边界可通过 lambda 给出域外状态，例如外流 `boundary=lambda coords,inside,axis,side,t: inside`，或给定解析状态 `boundary=lambda coords,inside,axis,side,t: exact(coords,t)`。`periodic` 可设为每方向 tuple，如 `(True,False)`，周期方向会绕过边界回调。

#### 四阶段三阶 SSPRK(4,3)

`ssprk43` 的阶段公式（`L=L(U,t)`）为：

```
u1 = u + dt/2 L(u,t)
u2 = u1 + dt/2 L(u1,t+dt/2)
u3 = 2/3 u + 1/3 u2 + dt/6 L(u2,t+dt)
u_next = u3 + dt/2 L(u3,t+dt/2)
```

阶段时间为 `t, t+dt/2, t+dt, t+dt/2`，边界回调使用各阶段时间。实现参考 [Gkeyll 官方 SSP 时间推进文档](https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html)。该方法时间三阶、SSP 系数为 2：如果空间离散的前向 Euler 步在 `dt_FE` 下具有强稳定性质，那么此 RK 可在 `dt<=2*dt_FE` 下继承该性质。这个条件不自动使无限制器的高阶 Euler FR 格式具有正性或 TVD 性质。当前默认 CFL 保持为 0.15。

`integrate` 接受固定 `dt`，也接受 `dt(U,t)` 回调；最后一步截断到终止时间。自定义 RK 可以传入 `method=lambda rhs,U,t,dt: ...`；`validate(U)` 可用于每步前和结束时验证状态，如 Euler 的 `e.valid`。

### 物理方程与通量选择

一阶双曲方程拆分在 `fr/equations/`，物理通量与数值界面通量独立选择：

| 方程 | 文件 | 状态变量顺序 | 工厂名称 |
|---|---|---|---|
| 非线性 Euler | `fr/equations/euler.py` | `rho, rho*u_1, ..., rho*u_d, E` | `euler` |
| 完整线化 Euler | `fr/equations/linearized_euler.py` | `rho', u'_1, ..., u'_d, p'` | `linearized-euler` |
| Maxwell | `fr/equations/maxwell.py` | `Ex, Ey, Ez, Hx, Hy, Hz` | `maxwell` |
| 等熵声学特例 | `fr/equations/acoustics.py` | `p', u'_1, ..., u'_d` | `acoustics` / `convected-acoustics` |

统一提供 `eq.flux(U,axis)`、`eq.speed(U,axis)`、`eq.valid(U)` 和 `eq.nvars`。这些是公共可调用函数，可直接传入 FR，也可用 lambda 包装。Euler 和线化 Euler 支持 `dim=1/2/3`；Maxwell 保留六个场分量，网格可只使用其中一至三个空间方向。各测例负责初始条件、状态布局、边界和物理参数。

```python
from fr import CartesianFR, make_equation, make_numerical_flux

# 分别选择任一方程；下面的网格示例使用线化 Euler
# eq = make_equation('euler', dim=2, gamma=1.4)
# eq = make_equation('maxwell', epsilon=1., mu=1.)
eq = make_equation('linearized-euler', dim=2,
                   mean=(.3,.2), rho0=1., p0=1/1.4, gamma=1.4)
mesh = CartesianFR((8,8), ((0.,1.),)*2, p=3,
                   flux=eq.flux,
                   numerical_flux=make_numerical_flux('rusanov',eq),
                   speed=eq.speed, periodic=True)
```

`make_numerical_flux(name,eq)` 支持 `central`、`rusanov`、`lax-wendroff`；提供特征绝对 Jacobian 的线化 Euler、声学、Maxwell 还支持 `roe` / `upwind`。目前非线性 Euler 未实现 Roe，选择该组合会明确报错。也可传入自定义 `lambda minus,plus,axis: ...`，原样返回。Lax–Wendroff 需要在实际时间步确定后调用 `prepare_step`，见下文工具说明。

完整线化 Euler 使用均匀背景 `rho0,p0,mean`，声速 `c0=sqrt(gamma*p0/rho0)`。独立的熵扰动 `rho' - p'/c0²` 和横向速度以背景流速对流；声学特例相当于令熵扰动为零。线化状态是扰动量，可为正或负。`from_acoustic` / `acoustic_state` 提供两种布局间的映射；线化方程的 `energy` 只统计声学能量，不包含独立熵扰动。

背景流 PML 测例支持直接选择完整线化 Euler：

```bash
python -m case.acoustic_mean_flow_pml --equation linearized-euler --validate
```

它仍使用同一个未经修改的 `SplitPML`。初始密度扰动取 `rho'=p'/c0²`，因此本脉冲应与等熵声学特例一致。测试另外验证了独立熵模态对流、线化通量与非线性 Euler 通量导数的一致性，以及旧导入路径兼容性。

含时薛定谔方程位于 `fr/equations/schrodinger.py`，通过 `Schrodinger(ldg, potential=...)` 或 `make_equation('schrodinger', ldg=..., potential=...)` 组装，状态为 `[Re psi, Im psi]`，直接提供 Hamiltonian 和演化右端；它使用下文的二阶 LDG 接口。

### 通量、边界与源项工具

各 case 自行传入框架回调，原有 lambda 接口继续可用。工具包括：

- `fr.fluxes.central`、`rusanov_flux(speed)`：中心与局部 Lax–Friedrichs 通量。
- `roe_flux(absolute_jacobian_action)`：传入 `abs(A) @ jump` 的方程实现；声学已提供精确特征分解。声学静止介质的 Roe 与特征迎风等价；这不是 Euler Roe 的实现。
- `upwind_flux(velocity)`：常速度标量平流迎风；系统使用特征分解通量。
- `LaxWendroffFlux(flux)`：Richtmyer 半步界面预测器，再求物理通量。与 FR + RK 组合后不等同于经典二阶有限体积 Lax–Wendroff 格式，也不保证任意 CFL 下稳定。通过 `integrate(..., prepare_step=...)` 在每个实际步长确定后调用 `face.prepare_step(dt, mesh.h)`，包含末步截断。
- `fr.boundaries.extrapolate`、`prescribed(state)`、`per_face({(axis,side): callback}, default=...)`、`acoustic_wall`：分别为外推、给定状态、逐面组合和声学刚壁。已有 Maxwell PEC 回调仍可直接组合；周期边界仍通过网格 `periodic` 指定。
- `CartesianFR(..., source=lambda coords,U,t: ...)`：在每个 RK 子步添加体源；`fr.pml.SplitPML` 提供通量、边界包装和阻尼源项，支持张量积 1D/2D/3D 布局。已有二维声学及下文 TMz 圆柱散射验证，尚未验证一般三维 Maxwell PML。

声学 case 可通过 `--flux central|roe|upwind|rusanov|lax-wendroff` 选择。其他 case 现有的默认通量和入口保持原有行为，需要时可导入这些工厂自主组合。

### Cartesian 二阶 LDG 算子

`fr/second_order.py` 中的 `CartesianLDG`，在现有 1D/2D/3D Cartesian FR 网格上实现两步 LDG：先求 `G(U)`，再求 `D(F)`。内部面采用互补交替迹 `U_hat=U_minus`、`F_hat=F_plus`。周期边界或齐次 Dirichlet 边界下，Gauss 加权内积满足 `D=-G*`，因而 `〈U,D G U〉=-||G U||²`。这是半离散性质；RK4 时间离散不严格保概率或能量。

梯度布局为 `[*单元维度,*解点维度,变量,空间方向]`，保留各方向梯度和变量间耦合，供后续 NS 黏性应力、热通量使用：

```python
from fr import CartesianLDG

ldg = CartesianLDG(s, boundary=lambda coords,U,axis,side,t: U*0)
grad = ldg.gradient(U, t)
# viscous_flux(coords,U,grad,t) 返回与 grad 同布局的通量张量。
viscous_rhs = ldg.diffusion(U, t, constitutive=viscous_flux)
```

这里的 `boundary` 返回实际 Dirichlet 边界值，而非 Euler 接口中的外部状态；散度在这些边界取内部通量。当前接口支持周期和 Dirichlet，尚未提供 Neumann/混合条件、曲边非结构网格上的二阶算子或完整 NS 方程及黏性壁面。一般本构回调的稳定性还需由具体物理模型保证，不能由标量扩散测试代替。

### 二维曲边与非结构网格

提供独立于 `CartesianFR` 的 `Mesh2D` / `MeshFR2D`。支持直边/曲边三角形、一般四边形及两者共形混合连接；单元无需正交、均匀或全局结构化编号。单元允许顺时针输入，读入时统一为逆时针。共享面自动建立邻接和积分点反向排列，材料及边界使用 Gmsh Physical 标签。现有笛卡尔测例保留原入口。

网格生成调用 PATH 中的 `gmsh` 可执行文件，不需要 venv 安装 Python gmsh、NumPy 或 meshio。已使用本机 Gmsh 4.12.1 实测；该版本在不规则四边形示例使用 Algorithm 8 时发生崩溃，示例已改用正常运行的 Algorithm 6 + Recombine。生成文件均保留 Physical 区域，读入器支持 **ASCII MSH 2.2 / 4.1**。支持完整 Lagrange 三角形和四边形，几何阶数 1–5；主要实测为一阶和三阶。二进制、serendipity 不完整单元、非 XY 平面网格和三维体单元会明确拒绝。要求共形连接，不支持悬挂节点、自适应非共形面或混合解阶；不负责修复这些网格。

#### 离散方法

依据 [Castonguay–Vincent–Jameson 的三角形 FR 论文](<doc/A New Class of High-Order Energy Stable Flux Reconstruction Schemes for Triangular Elements.pdf>) 第 5.4 节，首先采用 VCJH 的 **c=0，即 DG 成员**。三角形使用完整 `P_p` 多项式空间，四边形使用张量积 `Q_p`。`p=3` 时每个三角形为 10 个自由度，每个四边形为 16 个自由度；面上积分点数另行确定。

`fr/reference.py` 构造节点基、导数和积分规则；`fr/mesh.py` 管理网格、Gmsh 高阶节点排列与邻接；`fr/unstructured.py` 预积分每个曲边单元的物理质量矩阵、体积项和面提升矩阵。曲边 Jacobian、物理法向和边长因子均来自等参映射，并检查节点、边及积分点的 Jacobian 为正。三角形使用 Duffy 变换的 Gauss 积分，体积与曲边积分过积分，质量矩阵通过 Cholesky 求解。

这里实现的是 DG/FR-DG 成员；非线性通量采用积分点求值以减少欠积分误差，与论文中一般的节点通量插值不完全相同。**未实现 c>0 的完整 VCJH 参数族，也未把该文线性、直边条件下的稳定性证明推广宣称到任意曲边非线性问题。**

已验证：参考单元 SBP 恒等式、曲边自由流保持、仿射解析解、全局边界通量收支、封闭 PEC Maxwell 半离散能量、三角形平流加密收敛，以及真实 Gmsh 2.2/4.1 三次曲边文件。

#### 程序构造与回调接口

```python
from fr import Mesh2D, MeshFR2D, make_equation, make_normal_numerical_flux

# 也可 mesh = Mesh2D.read('grid.msh')
mesh = Mesh2D.from_cells(
    points=[(0,0), (1,0), (1.1,1), (0,1), (2,0), (2,1)],
    cells=[(0,1,2,3), (1,4,2), (4,5,2)],
    regions=[1,1,2],
)
eq = make_equation('linearized-euler', dim=2, mean=(.3,.2))
solver = MeshFR2D(mesh, p=3, flux=eq.flux,
                  numerical_flux=make_normal_numerical_flux('roe',eq),
                  speed=eq.normal_speed,
                  boundary=lambda coords, inside, normal, tag, t: inside)
# 状态布局为 (solver.ndof, eq.nvars)，solver.coords 为各自由度的 (x,y)
```

`Mesh2D.rectangle(nx,ny,kind='tri'|'quad'|'mixed',jitter=.2,seed=...)` 可直接构造抖动的不规则网格；`.mapped(lambda x,y: (X,Y), order=3)` 升阶几何后应用平滑映射。`write_svg` 输出实际采样的曲边网格预览。

公共物理通量仍为 `eq.flux(U,axis)`。不规则网格的数值通量则为 `callback(minus,plus,unit_normal)`，返回沿当前单元**外法向**的通量；`NormalFace` 包含状态、该法向的物理通量、坐标、区域和曲线标签。公共方程新增 `normal_speed`，`make_normal_numerical_flux` 支持中心、Rusanov 和已提供法向特征分解的 Roe/迎风（非线性 Euler 仍未实现 Roe）。Cartesian 的 Lax–Wendroff 接口不直接套用于一般曲面。

边界回调为 `boundary(coords,inside,unit_normal,physical_tag,t)`，可使用 `fr.boundaries.per_tag` 逐标签配置。体源为 `source(coords,U,t,region)`，在体积分点求值。时间推进继续使用公共 RK4 / SSPRK(4,3)。开放边界平流例使用随时间变化的解析外部状态，不使用周期面配对。

### 结果输出、云图与动画

直接运行算例就会在 `--output` 目录中生成图片，原有 CSV/PT/VTK 等输出继续保留。

| 算例 | 最后时刻 PNG | 非定常 GIF |
|---|---|---|
| `ringleb`、`prandtl_meyer` | 密度、压力、Mach 三幅云图 | 定常问题，不生成动画 |
| `isentropic_vortex` | 密度、压力、Mach | 三个物理量随时间演化 |
| `acoustic_pml`、`acoustic_mean_flow_pml` | 压力扰动 | 柱面波传播及 PML 吸收 |
| `maxwell_cavity` | Ez、电场模、磁场模 | 同一物理截面的驻波振荡 |
| `advection` | 标量场；1D 为曲线 | 1D/2D/3D 对流演化 |
| `mesh_advection` | 曲边混合网格上的标量云图 | 标量传播 |
| `schrodinger` | 一维概率密度曲线/二维概率密度云图及波函数实部 | 波包传播、势阱振荡和双缝干涉 |
| `mie_cylinder` | 瞬时散射 Ez，以及采样所得散射 Ez 相量幅值 | 瞬时散射 Ez 从启动到结束的演化 |

```bash
# 定常算例：最后一步的 PNG
python -m case.ringleb
python -m case.prandtl_meyer

# 非定常算例：最后一步 PNG + GIF；--fps 只影响播放速度
python -m case.isentropic_vortex --time 6 --frames 120 --fps 12
python -m case.acoustic_pml --time 3 --frames 60
python -m case.acoustic_mean_flow_pml --time 4 --frames 60
python -m case.maxwell_cavity --periods 3 --frames 120 --fps 12
python -m case.mesh_advection --kind mixed --time 2 --frames 120 --fps 12
python -m case.mie_cylinder --validate --frames 120 --fps 12
```

非定常算例默认 `--frames 60 --fps 12`；`--frames 0` 只生成最后一步 PNG，
`--no-plot` 完全关闭绘图（无需绘图库）。定常算例也支持 `--no-plot`。
网格生成工具本身没有物理场，不输出云图。

绘图在每个单元内独立插值再细分着色，保留 DG 单元间的不连续，不跨越曲边或材料界面连接插值。
三维场绘制物理域中间的 **z=(zmin+zmax)/2 的 xy 截面**，在该位置插值，不是沿 z 平均；截面位置写在图标题中。
声学移动坐标算例 `acoustic_pml` 的图中 x/y 表示随框架移动的计算坐标 xi/eta，
固定背景流算例则是实验室坐标。

动画使用整段动画统一的色标，正负扰动采用对称色标。初始状态、最后状态都会记录，
中间帧取接近均匀目标时间的首个已完成时间步，**不修改 RK 时间步或保存 RK 中间阶段**。
如果总时间步较少，实际帧数会少于 `--frames`；每帧显示真实物理时间。
GIF 按固定播放帧率播放，精确采样时间和色标写在对应的 `*_plot.json` 中。
收敛计算按网格分辨率分别输出图片和动画。PNG 表示指定结束时间的解，不额外宣称已达到稳态；
Mie 的相量幅值使用原有末尾周期采样结果，不能由单个瞬时值替代。

### 验证组织

`tests/` 覆盖参考单元恒等式、多项式精确性、自由流保持、面通量平衡、非自治时间推进、解析解收敛、边界条件、PML 参考对照、二阶算子伴随性质与双缝独立参考。最近一次完整回归共 **78 项测试通过**。以下算例中的误差表是各自列明参数下的实测结果。

## 算例

### 入口总览

| 测例 | 文件 / 模块 | 默认运行命令 | 默认输出 |
|---|---|---|---|
| 二维 Ringleb 亚声速流 | `case/ringleb.py` | `python -m case.ringleb` | `results/ringleb/` |
| 二维普朗特–迈耶膨胀扇 | `case/prandtl_meyer.py` | `python -m case.prandtl_meyer` | `results/prandtl_meyer/` |
| 二维等熵涡平移 | `case/isentropic_vortex.py` | `python -m case.isentropic_vortex` | `results/isentropic_vortex/` |
| 三维 PEC 长方体电磁驻波 | `case/maxwell_cavity.py` | `python -m case.maxwell_cavity` | `results/maxwell_cavity/` |
| 固定网格、非零 u/v 背景流 + 原 PML（实验） | `case/acoustic_mean_flow_pml.py` | `python -m case.acoustic_mean_flow_pml --validate` | `results/acoustic_mean_flow_pml/` |
| 二维柱面声脉冲 + 随流移动 PML | `case/acoustic_pml.py` | `python -m case.acoustic_pml --validate` | `results/acoustic_pml/` |
| Gmsh 贴体/不规则网格生成 | `case/mesh_generation.py` | `python -m case.mesh_generation` | `results/mesh/` |
| 二维三角形/四边形/混合曲边网格平流 | `case/mesh_advection.py` | `python -m case.mesh_advection --convergence` | `results/mesh_advection/` |
| 二维介电圆柱 TMz 散射 | `case/mie_cylinder.py` | `python -m case.mie_cylinder --validate` | `results/mie_cylinder/` |
| 1D/2D/3D 周期标量平流 | `case/advection.py` | `python -m case.advection --dim 3` | 终端指标 |
| 含时薛定谔五例 | `case/schrodinger.py` | `python -m case.schrodinger --case all` | `results/schrodinger/` |
| 双缝独立精度对照 | `case/double_slit_accuracy.py` | `python -m case.double_slit_accuracy --time-check` | `results/double_slit_accuracy/` |

双曲算例通常采用 SSPRK(4,3)，部分入口可用 `--rk rk4` 切换；薛定谔五例使用 RK4，双缝独立验证使用四阶对称分裂。每个入口均可通过 `--help` 查看参数。

### 周期标量平流（1D / 2D / 3D）

周期标量平流用于验证 Cartesian 1D/2D/3D 通量接口、时间推进与全局质量守恒。

```bash
python -m case.advection --dim 1
python -m case.advection --dim 2
python -m case.advection --dim 3
```

三维示例推进至 `t=0.1` 的质量变化约 `-1.78e-15`。非定常输出包含最终场和动画，三维绘制域中心的 xy 截面。

### Ringleb 亚声速流

```bash
python -m case.ringleb --n 8 --p 3 --time .1
python -m case.ringleb --n 4 --p 3 --time .1 --convergence
python -m case.ringleb --n 8 --p 2 --time 6 --perturb .001 --output results/relaxation
```

第一版计算区域为 `[-0.5,0] × [1.5,2]`，是 Ringleb 流的光滑亚声速矩形子区域，四边使用精确外部状态与内部状态计算 Rusanov 通量。**尚未实现完整弯曲流道、滑移壁面和跨声速区域。** 此设置用于先验证论文的空间离散，不能视为完整流道基准复现。

取速度大小 `q`、声速 `c=sqrt(1-q²/5)`、密度 `rho=c^5`，定义

```
J = 1/c + 1/(3c³) + 1/(5c⁵) - 0.5 log((1+c)/(1-c))
x = (1/q² - 2/k²)/(2rho) + J/2
y = sqrt(1-q²/k²)/(k rho q)
u = q sqrt(1-q²/k²), v = q²/k
p = rho c²/gamma
```

精确解通过 `(x-J/2)²+y²=1/(4rho²q⁴)` 的二分反演得到，限定上述子区域所在的分支。公式与符号约定参考 [Nektar++ RinglebFlow 官方实现](https://doc.nektar.info/doxygen/5.1.0/_ringleb_flow_8cpp_source.html)，这里独立用标量二分算法实现，并以物理坐标有限差分验证 Euler 通量散度为零。

默认从精确解的解点采样出发，推进至有限时间，测量离散化产生的误差。`--perturb 0.001` 会为所有守恒量乘上内部正弦扰动；它可检查恢复过程。运行到指定时间即结束，不代表自动达到定常残差阈值。

#### 二次多项式验证

原始 RK4 结果，`p=2, t=0.1, CFL=0.15`，CPU float64：

| 网格 | 密度 L2 误差 | 加密收敛阶 |
|---|---:|---:|
| 4 × 4 | 3.20090e-6 | — |
| 8 × 8 | 4.69594e-7 | 2.769 |
| 16 × 16 | 6.14834e-8 | 2.933 |

与一般光滑 Euler 解的 `p+1=3` 阶相符；不将论文中标量线性问题的超收敛阶套用于 Euler。

四项自动验证：多项式求导和 Radau 积分性质、精确解的连续 Euler 方程残差、全局离散守恒与面通量平衡、Ringleb 网格收敛。当前没有激波限制器、正性保持器或非线性去混叠；非物理密度/压力会报错。

扰动恢复实测：`n=8, p=2, perturb=0.001, t=6`，共 9033 步。密度误差从初始 `4.19015e-4` 降至 `5.39234e-7`，最终密度半离散残差 `2.33825e-9`（四个守恒量残差均低于 `5.2e-9`）。输出见 `results/relaxation/`。

#### 三次多项式验证

默认 `p=3`：每方向 4 个 Gauss 解点，每单元 16 个解点，预期四阶精度。

```bash
python -m case.ringleb --n 4 --p 3 --time 0.1 --convergence --output results/p3
```

同一算例使用原始 RK4 的实测结果：

| 网格 | 密度 L2 误差 | 加密收敛阶 |
|---|---:|---:|
| 4 × 4 | 2.22193e-8 | — |
| 8 × 8 | 1.47971e-9 | 3.908 |
| 16 × 16 | 9.21734e-11 | 4.005 |

结果数据及图像见 `results/p3/`。前文 `--p 2` 命令保留作为二次多项式对照。

默认 SSPRK(4,3) 的同组 p=3 密度 L2 误差为 `2.22193e-8 / 1.47971e-9 / 9.21791e-11`，观测阶为 `3.9084 / 4.0047`，见 `results/p3_ssprk43/`。此定常解主要验证空间收敛；时间三阶另由非自治 ODE 测试验证。

默认输出 `metrics.json`、坐标/守恒量/精确解 `.pt`、逐解点 `.csv`、`.svg` 和 PNG。守恒量误差顺序为 `[rho,rho*u,rho*v,E]`，L2 使用物理面积归一化 Gauss 权重；残差为同一范数下的半离散 `dU/dt`。SVG 是解点采样色块，PNG 在单元内部插值。

### 普朗特–迈耶膨胀波

`case/prandtl_meyer.py`：`PrandtlMeyerFan` 为解析解与参考状态，`PrandtlMeyerCase` 通过已有 `CartesianFR`、Euler、Rusanov 及 SSPRK(4,3) 接口组装算例；没有修改空间离散或添加源项。可从 `case.prandtl_meyer` 导入两个类。

默认来流 `M1=2.5`、`gamma=1.4`、`rho1=p1=1`，流向沿正 x；从原点绕凸角顺时针转向 `15°`。参数选用 [NASA NPARC 的 Mach 2.5 / 15° 验证案例](https://www.grc.nasa.gov/www/wind/valid/pm15/pm15.html)；此处仅采用其无量纲参数和解析参考关系，并未复现 NASA 的拐角网格、边界条件或数值方法。该膨胀满足等熵关系，马赫数增加、静压下降，参考 [NASA 膨胀扇说明](https://www.grc.nasa.gov/WWW/BGH/expans.html)。

```bash
source venv/bin/activate
# 包含上游常状态、整个膨胀扇和下游常状态的矩形区域
python -m case.prandtl_meyer --n 16 --p 3 --time .1
# 完整区域的网格加密（8、16、32 单元/方向）
python -m case.prandtl_meyer --n 8 --convergence --output results/prandtl_meyer/full
# 扇内光滑区域的高阶精度验证（4、8、16 单元/方向）
python -m case.prandtl_meyer --region smooth --n 4 --time .01 --convergence --output results/prandtl_meyer/smooth
# 可调来流 Mach、转角、gamma 和 RK
python -m case.prandtl_meyer --mach 2 --turn 10 --gamma 1.4 --rk rk4
```

与现有 Ringleb 验证类似，初值为解析解的解点采样，四边使用解析外部状态，推进到指定时间后比较误差。默认 `full` 区域为 `[0.5,1.5] × [0,0.69829725]`，位于真实流体区域内部，跨越扇头和扇尾，但**不包含原点尖角或真实拐角壁面，也不通过壁面转角从均匀来流生成膨胀扇**。边界与初值定义的是解析场保持/离散误差验证；运行到 `--time` 不表示达到定常残差阈值。

`--region smooth` 会根据扇头、扇尾自动选取完全位于膨胀扇内的小矩形。完整扇的状态连续，但扇头/扇尾的一阶导数发生跳变；无限制器 FR 在这些位置可能有小幅振荡，完整域误差不应要求四阶收敛。光滑区域用于单独检查 `p=3` 的四阶空间精度。默认 CFL 仍为 0.15，未增加限制器。

解析场的构造（角度均为弧度）：

```
nu(M) = sqrt((gamma+1)/(gamma-1)) atan(sqrt((gamma-1)/(gamma+1)*(M²-1)))
        - atan(sqrt(M²-1))
nu(M2) = nu(M1) + delta
mu(M) = asin(1/M)
phi = atan2(y,x)
head = mu(M1), tail = mu(M2)-delta
```

扇内通过单调二分反演 `phi = mu(M) - [nu(M)-nu(M1)]`；流向角 `theta = -[nu(M)-nu(M1)]`。`phi>=head` 使用上游常状态，`phi<=tail` 使用下游常状态。温度比 `T/T1 = (1+(gamma-1)M1²/2)/(1+(gamma-1)M²/2)`；密度、压力按等熵关系恢复，再生成 Euler 守恒量。解析函数限定 `x>0`、点在转向壁面以上。

默认参数的参考值：

| 量 | 解析值 |
|---|---:|
| 下游 Mach `M2` | 3.23684311 |
| `p2/p1` | 0.32742741 |
| `rho2/rho1` | 0.45045727 |
| `T2/T1` | 0.72687786 |
| 扇头角 | 23.57817848° |
| 扇尾角 | 2.99554195° |

这些值与 NASA 验证页 Table 7 的四位小数参考值一致。等熵常量、总焓、扇边界连续性和物理坐标 Euler 通量散度均另有测试。

光滑扇内，`p=3, t=0.01, SSPRK(4,3)` 的密度误差为：

| 网格 | 密度 L2 | 加密阶 |
|---|---:|---:|
| 4 × 4 | 3.90162e-8 | — |
| 8 × 8 | 2.90278e-9 | 3.749 |
| 16 × 16 | 2.00993e-10 | 3.852 |

默认完整扇 `p=3,t=0.1`，8×8 和 16×16 网格的 Mach L2 误差分别为 `2.35026e-3`、`7.04914e-4`；两个网格的密度、压力均保持正值。

输出同样包含 `metrics.json`、`.pt`、`.csv` 和采样色块 `.svg`（密度、Mach、密度绝对误差）。CSV 保存实际物理坐标；`.pt` 同时保存数值解与解析解。报告中的 `reference` 是解析上下游状态，`mach_l2` 是数值与解析 Mach 的加权误差，`max_entropy_error` 是数值 `p/rho^gamma` 相对参考常量的最大绝对偏差。绘图导出现在读取算例的 gamma 和区域范围，不再固定为 Ringleb 参数。

全部 18 项测试通过，其中新增 7 项覆盖 NASA 参考值、等熵/总焓关系与扇边连续性、连续 Euler 残差（两种 gamma）、光滑区域四阶趋势、完整扇的有限时间计算、非法参数以及自定义 gamma 的 CSV/SVG 输出。

完整扇第三组 `32×32` 网格的 Mach L2 误差为 `2.98998e-4`，最小密度 `0.449779`、最小压力 `0.326737`，结果图见 `results/prandtl_meyer/full/pm_full_n32_p3.svg`。

### 二维等熵涡平移

`case/isentropic_vortex.py`：`IsentropicVortex` 提供随时间平移的解析场，`IsentropicVortexCase` 组装二维 Euler、双周期边界、Rusanov 面通量和 SSPRK(4,3)。两者可从 `case.isentropic_vortex` 导入。公式采用标准 Shu 等熵涡，参考 [HyPar 官方算例](https://hypar.github.io/a00160.html)；周期盒需要足够大以忽略邻域涡的影响，参见 [PLUTO 官方说明](https://plutocode.ph.unito.it/Doxygen/Test_Problems/_h_d_2_isentropic___vortex_2init_8c.html)。

默认 `gamma=1.4`、涡强度 `beta=5`、背景速度 `(1,1)`、初始涡心 `(0,0)`，远场 `rho=p=T=1`。计算域为 `[-10,10]²`、`p=3`，每单元 4×4 个 Gauss 解点。

```bash
source venv/bin/activate
python -m case.isentropic_vortex --n 16 --p 3 --time .1
python -m case.isentropic_vortex --n 8 --convergence
# 细网格结果
python -m case.isentropic_vortex --n 64 --time .1 --output results/isentropic_vortex/fine
# 调整速度、涡强度、初始中心、计算域及 RK
python -m case.isentropic_vortex --velocity 1 0 --beta 3 --center 0 0 --length 20 --rk rk4
# 穿过周期边界：涡心从 (9.95,9.95) 移至 (-9.95,-9.95)
python -m case.isentropic_vortex --center 9.95 9.95 --time .1 --output results/isentropic_vortex/crossing
# 默认速度下一个完整绕行周期为 20，可用 --time 20 计算
```

设 `X=x-x0-u0*t`、`Y=y-y0-v0*t`、`r²=X²+Y²`，解析解为

```
u = u0 - beta/(2pi) exp((1-r²)/2) Y
v = v0 + beta/(2pi) exp((1-r²)/2) X
T = 1 - (gamma-1) beta²/(8 gamma pi²) exp(1-r²)
rho = T^(1/(gamma-1)), p = rho^gamma
E = p/(gamma-1) + rho*(u²+v²)/2
```

周期盒中使用最近镜像距离 `X,Y ∈ [-L/2,L/2)`，使涡心穿越周期边界时解析参考场正确跟随。无限平面高斯涡并不是严格光滑的有限盒周期解；最近镜像的拼接处仍有极小的高斯尾差异。默认 `L=20` 时，半盒距离处的速度扰动约 `1e-21`，低于 float64 有效精度。改用较小的盒子（例如文献中常见的 `L=10`）时，应考虑尾部拼接误差，不能将任意小盒都视为严格的光滑周期基准。

这是**非定常平移**算例：初值保存为 `case.initial`，解析场由 `case.exact_at(t)` 获取；`case.run()` 结束后 `case.exact` 更新为终止时刻的参考场，CSV/SVG 的误差和 `.pt` 中的 `exact` 均与该时刻对应。不会拿平移后的数值解与初值直接相减。无源项，也不对 Euler 残差作解析抵消。

输出与已有算例一致，另在 `.pt` 中保存 `initial`。`metrics.json` 报告四个守恒量的 L2 误差、周期涡心位置、最小密度/压力、最大等熵偏差 `|p/rho^gamma-1|`，以及质量、两个动量和总能量的初末积分及绝对变化。`L2` 使用面积归一化的 Gauss 加权范数。

CPU float64、`p=3,t=0.1,CFL=0.15,SSPRK(4,3)` 实测结果：

| 网格 | 密度 L2 误差 | 加密阶 |
|---|---:|---:|
| 8 × 8 | 1.300925e-03 | — |
| 16 × 16 | 1.080909e-04 | 3.589 |
| 32 × 32 | 1.090083e-05 | 3.310 |
| 64 × 64 | 6.890070e-07 | 3.984 |

细网格达到约四阶空间收敛；64×64 网格中四个守恒量的积分变化最大约 `3.64e-12`（总能量约 `1394.76`）。汇总指标见 `results/isentropic_vortex/convergence.json`，结果图见 `results/isentropic_vortex/fine/vortex_n64_p3.svg`。

### 三维 PEC 腔体电磁驻波

通用方程位于 `fr/equations/maxwell.py`（兼容 `fr.maxwell` 和 `from fr import Maxwell`）和算例 `case/maxwell_cavity.py`。算例实际使用三维 FR、长方体六个物理壁面和随时间振荡的解析本征模，不是用解析外部状态覆盖边界。当前介质为均匀、各向同性、无损材料，无源、无电荷；不包含介质界面、导电率或散度清理。

默认配置：边长 `(1,1.3,1.7)`、模态指标 `(1,1,1)`、`epsilon=mu=1`（归一化波速 `c=1`）、每方向 4 个单元、三次多项式、每单元 `4³=64` 个解点、中心面通量和 SSPRK(4,3)，计算一个完整振荡周期。保留现有 RK，不对其耗散作额外修正。

```bash
source venv/bin/activate
# 默认中心通量：一个完整周期
python -m case.maxwell_cavity --n 4 --periods 1 --flux central
# 同一个算例改用 Roe 特征通量
python -m case.maxwell_cavity --n 4 --periods 1 --flux roe --output results/maxwell_cavity/roe_period
# 自定义长方体、各向单元数和模态
python -m case.maxwell_cavity --lengths 1 1.3 1.7 --cells 4 6 8 --mode 1 1 2
# 空间加密，固定物理时间（--time 优先于 --periods）
python -m case.maxwell_cavity --n 2 --time .1 --convergence --flux central --output results/maxwell_cavity/central
python -m case.maxwell_cavity --n 2 --time .1 --convergence --flux roe --output results/maxwell_cavity/roe
# 均匀介质参数和 RK 均可配置
python -m case.maxwell_cavity --epsilon 2 --mu 1 --rk rk4
```

状态为 `U=(Ex,Ey,Ez,Hx,Hy,Hz)`，求解

```
epsilon * dE/dt = curl(H)
mu      * dH/dt = -curl(E)
```

沿正坐标方向 `e_a` 的守恒通量为 `F_a=(-e_a × H / epsilon, e_a × E / mu)`，特征速度为 `-c,0,+c`，其中 `c=1/sqrt(epsilon*mu)`。

#### 中心 / Roe 面通量与 PEC 壁面

中心通量为 `F*=(Fminus+Fplus)/2`。该常系数、Gauss 配点 FR-DG 格式配 PEC 镜像边界，在半离散层面保持离散电磁能量；时间推进的误差仍存在。

Roe 选项为这个常系数 Maxwell 系统的精确特征迎风通量：

```
F* = (Fminus+Fplus)/2 - c/2 * P_t * (Uplus-Uminus)
```

`P_t` 保留切向 E/H 分量、去掉两个法向分量，对应 `|A_a|=c P_t`。因此它与对全部六个分量施加耗散的标量 Rusanov 不同；**Roe 本身仍属于迎风通量**。中心通量默认无空间耗散，Roe 提供可选耗散。

算例同样支持自定义 lambda：

```python
from case.maxwell_cavity import MaxwellCavityCase
case = MaxwellCavityCase(
    n=4, p=3,
    numerical_flux=lambda minus, plus, axis: .5*(minus.flux+plus.flux),
)
U, metrics = case.run(periods=1)
```

PEC 域外镜像状态取 `E_t^+=-E_t^-`、`E_n^+=E_n^-`、`H_t^+=H_t^-`、`H_n^+=-H_n^-`。公共平均迹满足切向 `E=0`；对这个无初始法向磁通的腔体模态也满足法向 `B=0`。这与 [Maxwell DG 的 PEC 镜像条件](https://publikationen.bibliothek.kit.edu/1000033941/2494275) 一致。壁面由镜像状态和所选面通量弱施加，单侧单元插值迹在有限网格上不必严格等于零。

#### 解析模态与诊断

本例限制三个模态指标均为正整数，以确保默认测试包含三个方向的空间变化；没有实现含零指标的全部 TE/TM 模态族。定义 `k_a=pi*m_a/L_a`，选取归一化极化向量 `A` 满足 `k·A=0`（代码取与 `(1,1,-(kx+ky)/kz)` 平行的向量），则

```
E0 = (Ax cos(kx*x) sin(ky*y) sin(kz*z),
      Ay sin(kx*x) cos(ky*y) sin(kz*z),
      Az sin(kx*x) sin(ky*y) cos(kz*z))
omega = c sqrt(kx²+ky²+kz²)
H0 = -curl(E0)/(mu*omega)
E(x,t) = E0(x) cos(omega*t)
H(x,t) = H0(x) sin(omega*t)
```

该场满足两条旋度方程、两个连续散度约束及全部 PEC 边界。解析电磁能量为 `epsilon*volume*amplitude²/16`。默认频率 `f=omega/(2pi)=0.6960130634`，周期 `T=1.4367546424`。

输出文件：

- `.pt`：完整三维坐标、六分量数值场、初值、终止时刻解析场、诊断和采样历史。
- `.vts`：标准 VTK StructuredGrid，可用 ParaView 查看三维 E/H 向量、解析场、场误差和能量密度。这里保存的是 Gauss 解点采样，不含箱壁点或单元间双侧面迹。
- `*_history.csv`：时间、电磁总能量、相对能量变化、电场/磁场的模态投影系数及对应解析 cos/sin，用于检查驻波振荡与能量交换。
- `metrics.json`：六分量 L2、总场误差、E/H 分别的误差、能量、散度诊断、单侧壁面迹误差和频率估计。

`field_l2` 是六分量体积归一化 L2 范数的平方和开方，用于默认归一化参数下的网格比较。`semidiscrete_energy_rate` 独立计算空间算子的 `dW/dt`，不含 RK；总能量变化包含时间离散影响。`measured_frequency` 从两组模态投影的展开相位拟合得到，不是复制解析频率；只有采样数足够时报告，可通过 `--sample-every` 控制采样间隔（应足够密，避免相位混叠）。

`broken_divD_l2`、`broken_divB_l2` 只测量各单元内部的多项式散度，不含界面法向跳跃的分布项，不应当作全局散度约束已严格满足的证明。初始解析场采样也可能有非零的离散散度，因此同时记录初始值。当前没有增加散度清理。

#### 验证结果

CPU float64，`p=3,t=0.1,CFL=0.15,SSPRK(4,3)`：

| 网格 | 中心通量总场 L2 | 加密阶 | Roe 总场 L2 | 加密阶 |
|---|---:|---:|---:|---:|
| 2³ | 1.13417e-3 | — | 6.07434e-4 | — |
| 4³ | 1.60277e-4 | 2.823 | 4.77759e-5 | 3.668 |
| 8³ | 1.94160e-5 | 3.045 | 3.17499e-6 | 3.911 |

此设置中，三次多项式的中心通量场误差约三阶，Roe 约四阶；不能直接将之前 Euler/Rusanov 的四阶结果套到中心 Maxwell 通量。两种通量的误差均随加密下降。

`4³` 网格使用中心通量计算完整一个周期：总场 L2 误差 `1.16314e-4`，相对能量变化 `-2.56846e-7`，模态投影测得的相对频率误差 `2.89099e-7`；半离散能量变化率约 `-3.82e-19`。初始能量与解析能量均为 `0.138125`。该次实算结果位于 `results/maxwell_cavity/period/`。

### 二维柱面声脉冲与随流移动 PML

```bash
source venv/bin/activate
python -m case.acoustic_pml --n 16 --p 3 --u0 .3 --v0 .2 --time 3 --validate
python -m case.acoustic_pml --flux central
python -m case.acoustic_pml --flux upwind
python -m case.acoustic_pml --flux lax-wendroff
```

均匀背景流默认取 `(u0,v0)=(0.3,0.2)`。初始压力扰动为 `exp(-(xi²+eta²)/(2*0.15²))`，初始扰动速度为零；随后压力梯度产生两个方向的扰动速度。没有持续点声源。线性声学方程使用 `rho=c=1`，压力幅值可任意缩放；初值是圆对称脉冲，包含向外传播的柱面声波，并非单频、纯外传解析波。

**坐标与适用范围：** 使用随均匀背景流平移的坐标 `xi=x-u0*t, eta=y-v0*t`，在此坐标中求解

```text
p_t + rho*c²*(u'_xi+v'_eta) = 0
u'_t + p_xi/rho = 0
v'_t + p_eta/rho = 0
```

这等价于实验室坐标中带均匀对流的线性声学方程。PML 与计算区域随背景流一起移动；**尚未实现固定实验室边界、斜向平均流的稳定 PML**，不能用本例宣称验证了该问题。CSV 同时输出计算坐标、物理坐标、扰动速度和总速度；波形在计算坐标中保持圆对称，在实验室坐标中同时沿 x、y 平移。

默认计算域 `[-2,2]²`，无阻尼内部区域 `[-1,1]²`。`SplitPML` 将状态拆为每方向一份，物理量为各份之和；各方向通量只进入对应分量，并添加 `-sigma_axis*Q_axis`。阻尼在内部为零，向外按三次多项式增长，角区两个方向同时阻尼；外壁仍为刚壁。`reflection=1e-6` 是连续层的剖面设计参数，不是离散反射率保证。该工具适用于零平均流、常系数系统；不能直接给含平均流的方程加此阻尼并认为仍然完美匹配。

`--validate` 另算相同单元尺寸的扩大域无阻尼参考解，以及原大小区域的刚壁解，比较内部区域的物理状态 L2 误差。扩大域保证在计算结束前外边界无法影响内部。参考仍为数值解，测试衡量吸收层引入的误差，而非解析解精度；验证中的各区域采用相同时间步，以减少时间离散差异。输出 `metrics.json` 与 `field.csv`。

一次 `--n 16 --p 3 --time 3 --validate` 的结果：PML 内部状态误差 `1.18e-5`，刚壁误差 `1.27e-1`，比值 `9.28e-5`。这是此脉冲、网格和终止时间下的比较指标。

#### PML 剖面与层厚对照

保持内部单元尺寸 `h=0.25`、多项式阶数和初始脉冲相同，可以比较：

```bash
# 原配置：每边 PML 宽 1，4 层单元
python -m case.acoustic_pml --n 16 --pml-thickness 1 --time 4 --validate
# 加厚：每边 PML 宽 2，8 层单元；扩大外域以保持内部网格尺寸
python -m case.acoustic_pml --n 24 --pml-thickness 2 --time 4 --validate
```

`--pml-thickness` 同时设置层宽与外域半宽 `1+thickness`；`--n` 为整个区域每方向的单元数。加厚时应相应增加单元数，避免把内部声波网格也变粗。`--reflection` 控制阻尼剖面的理论衰减目标，`--pml-order` 控制渐变阶数；目标数值更小会增加阻尼强度，并不保证离散反射更小。优先加厚、保持平滑过渡，再检查网格加密效果。

验证计算使用相同名义时间步（含相同末步截断）。`core_pressure_max_error`、`core_pressure_error_over_initial_peak` 和 `core_error_energy_over_initial`，分别报告终止时刻内部压力最大误差、该误差与离散初始峰值的比值、误差场声能与初始声能的比值。误差均相对于扩大域参考解；内部残余声能本身包含真实二维波尾，不能直接视为反射能量。上述指标也不是通过入射/反射波分离得到的反射系数。

固定 `h=0.25, p=3, dt=0.003, t=4` 的剖面对比（共同参考域 `[-7,7]²`）：

| 每边层宽 | 设计参数 reflection | 内部压力最大误差 | 误差声能 / 初始声能 |
|---|---|---|---|
| 1 | 1e-6 | 1.061e-5 | 1.208e-9 |
| 1 | 1e-9 | 1.075e-5 | 1.232e-9 |
| 2 | 1e-6 | 5.858e-6 | 4.735e-10 |

原配置误差已很小。更强阻尼没有改善这一网格下的误差；加厚可以进一步降低该终止时刻的误差。数据保存在 `results/acoustic_pml_tuning/sweep.json`，复现脚本可用 `PYTHONPATH=. python results/acoustic_pml_tuning/reproduce.py` 运行。以上是终止时刻误差，不是时域最大反射率；加厚也推迟了外壁返回时间，长期反射仍需延长计算验证。

### 固定网格背景流与现有 PML

`case/acoustic_mean_flow_pml.py`。本例直接把现有 `SplitPML` 用于带背景对流的声学通量。阻尼源项、剖面、方向拆分和外边界包装均沿用现有实现。

```bash
source venv/bin/activate
# 默认 u0=0.3、v0=0.2，每边 8 层 PML，p=3，SSPRK(4,3)
python -m case.acoustic_mean_flow_pml --validate
# 自行选择两个背景速度分量
python -m case.acoustic_mean_flow_pml --u0 .4 --v0 -.2 --validate
# 低成本较长时间检查；其网格比默认更粗、PML 更薄
python -m case.acoustic_mean_flow_pml --n 12 --pml-thickness 1 --time 12 --output results/acoustic_mean_flow_pml_long
```

固定实验室坐标中的线性方程为：

```text
D = partial_t + u0*partial_x + v0*partial_y
D p' + rho*c²*(partial_x u' + partial_y v') = 0
D u' + partial_x p'/rho = 0
D v' + partial_y p'/rho = 0
```

`ConvectedAcoustics` 的各向通量为 `F_axis(U) = mean_axis*U + F_acoustic_axis(U)`。Roe/特征迎风使用特征速度 `mean_axis-c, mean_axis, mean_axis+c`，包括横向扰动的对流模态；CFL 也包含背景速度。这里的变化属于物理方程与相应通量，并未修改 PML。初始压力仍为宽度 0.15 的圆对称高斯脉冲，初始扰动速度为零；总初始速度为 `(u0,v0)`。波包在固定网格中既扩散又被背景流携带。外壁对扰动法向速度反射，其作用是作为吸收层后的截断；无 PML 的同域解是相同外壁的对照。

默认 `n=24`、外域 `[-3,3]²`、内部 `[-1,1]²`，PML 每边宽 2，保持单元尺寸 0.25。支持与原声学例相同的 `--flux`、`--reflection`、`--pml-order`、`--pml-thickness`、`--dt` 参数。

`--validate` 使用相同方程、相同单元尺寸、相同时间步的扩大域参考解，并按最大对流声速扩大参考域。报告内部参考误差、同域反射壁误差及误差能量。`metrics.json` 还记录全程最大能量比、终态能量比及有限性；`history.csv` 保存每个已接受状态的能量和压力峰值，`field.csv` 输出固定物理坐标及扰动/总速度。旧 `case.acoustic_pml` 的平移坐标例仍保留，两者可独立运行。

这是“把已有 PML 直接用于含流方程”的数值实验，未提供任意流速或长时间的完美匹配、稳定性保证。个别参数下的吸收效果应以本例实际计算为准。

已运行的粗网格检查：`n=12, p=3, thickness=1, (u0,v0)=(0.3,0.2)`，`t=3` 时内部最大压力误差相对初始离散峰值为 `5.93e-5`，内部状态误差为反射壁对照的 `2.68e-3`。同一配置运行至 `t=12` 时所有状态有限，监测的总声能从未超过初始值，终态总声能 / 初始声能为 `3.94e-5`；这是有限时间观测，不能代替一般稳定性证明。

默认加厚配置 `n=24, thickness=2, (u0,v0)=(0.3,0.2), t=4` 也已运行：内部最大压力误差 / 初始离散峰值为 `2.38e-5`，误差声能 / 初始声能为 `4.54e-9`，监测期间总声能未超过初始值。完整输出位于 `results/acoustic_mean_flow_pml/`。

也可通过 `--equation linearized-euler` 选择完整线化 Euler；初始 `rho'=p'/c0²` 对应等熵声学脉冲。

### Gmsh 网格生成与曲边混合网格平流

网格接口和离散方法见“框架设计”。本例生成网格或读入网格后求解开放边界平流；网格生成本身不输出物理云图。

```bash
source venv/bin/activate
# Gmsh 多块结构化、三次曲边、介电圆柱内外共形四边形网格
python -m case.mesh_generation --kind cylinder-quad --order 3 --output results/mesh/cylinder_quad.msh
# Gmsh 自动生成的无结构三角形圆柱网格
python -m case.mesh_generation --kind cylinder-tri --order 3 --output results/mesh/cylinder_tri.msh
# 一般不规则四边形，导出旧版 ASCII MSH 2.2
python -m case.mesh_generation --kind irregular-quad --version 2.2 --output results/mesh/irregular.msh

# 实际读入 Gmsh 网格运行高阶平流
python -m case.mesh_advection --mesh results/mesh/irregular.msh
# 程序生成抖动、曲边网格，并逐级加密
python -m case.mesh_advection --kind tri --n 4 --convergence
python -m case.mesh_advection --kind quad --n 4 --convergence
python -m case.mesh_advection --kind mixed --n 4 --convergence
```

不规则混合网格 `p=3, t=.05` 的实测加密结果：

| 基础每边单元数 | 三角形/四边形合计 | 状态 L2 误差 | 观测阶 |
|---|---:|---:|---:|
| 4 | 24 | 2.844e-4 | — |
| 8 | 96 | 3.240e-5 | 3.13 |
| 16 | 384 | 2.144e-6 | 3.92 |

### 贴体介电圆柱 TMz 散射

`case/mie_cylinder.py` 求解二维无限长介电圆柱的 **TMz** 散射，即未知量 `(Ez,Hx,Hy)`；三维球体 Mie 散射不在本例范围。`MaxwellTMz` 是公共 Maxwell 的投影，正常入射平面波沿 +x 传播。默认半径 `.5`、波长 `2`、柱内相对介电常数 `4`、相对磁导率 `1`，外部真空；界面连续条件通过物理单位法向的阻抗 Riemann 通量处理。

默认自动生成 176 个三次曲边四边形：中心块 + 内部环形块 + 外部环形块，材料界面贴合圆周；也可读入三角形及混合网格。几何仍是高阶多项式圆弧逼近，不是精确有理圆弧，但不再使用阶梯圆柱。自备网格需覆盖相同区域，Physical Surface 1 为外介质、2 为柱内，接口标签 2 可用于几何误差检查；外边界标签 1。

```bash
python -m case.mie_cylinder --validate
# 已有曲边三角形网格
python -m case.mie_cylinder --mesh results/mesh/cylinder_tri.msh --validate --output results/mie_cylinder_tri
# 增加网格密度；延长预热以检查稳态收敛
python -m case.mie_cylinder --angular 6 --radial 12 --warmup-periods 12 --validate --output results/mie_cylinder_refined
```

求解散射场，入射波为光滑开启的单频平面波。材料容量形式为 `diag(epsilon,mu,mu) U_t + div(F_bare)=0`，材料界面两侧使用各自阻抗；散射场体源为 `(I-C) dU_inc/dt`，求质量方程后除以材料容量。无材料反差时散射源严格为零，已加入测试。圆柱之外复用现有 Cartesian 方向分裂 `SplitPML`，通过曲边积分变换到物理坐标；无阻尼区为 `[-1.5,1.5]²`，外域 `[-2.5,2.5]²`。PML 吸收散射场，解析入射场不在层内衰减，外截断为 PEC。

参考解在 `case/cylinder_exact.py` 中通过 Fourier–Bessel / Hankel 圆柱级数独立计算，约定 `exp(-i omega t)`，出射波为 `H^(1)`；这是均匀、无损圆柱的中等尺寸参数算例，当前参考实现并非针对任意大参数的通用特殊函数库。默认前 6 个周期预热，再对后 2 个周期做复振幅积分；相邻周期差用于检查残余瞬态。数值远场用内部无阻尼环带的出射柱面波系数拟合获得（角向模态 -6..6），并与级数角分布及总散射宽度对照。

已实算的默认四边形结果：散射 Ez 相对 L2 误差 **0.9468%**，总散射宽度误差 **0.2901%**，角分布相对 L2 误差 **0.8964%**。总散射宽度解析值 `3.983545`，数值拟合值 `3.995101`。此时相邻周期场变化仍为 `1.28%`，精度受残余瞬态、网格、PML 和远场拟合共同影响，不能据此宣称稳态网格收敛。

输出包括 `mesh.svg`、`field.csv`（散射 Ez 复振幅）、`history.csv`、`scattering_width.csv`（解析/数值角分布）、`metrics.json`、`solution.pt`。默认自动生成时还保留 `.geo` 和 `.msh`，方便用 Gmsh 检查几何及区域标签。

另已用 Gmsh 生成的 **282 个无结构三次曲边三角形**（基础尺寸 `h=.5`）完成相同 8 周期计算：散射 Ez 相对 L2 误差 **0.8436%**，总散射宽度误差 **0.0980%**，角分布相对 L2 误差 **0.8307%**。输出保存在 `results/mie_cylinder_tri/`。该网格与默认四边形网格的分辨率和形状不同，这两组数据用于验证两种单元都能完成散射求解，不构成严格的网格收敛对比。

### 含时薛定谔方程：五个经典算例

`fr/equations/schrodinger.py` 实现无量纲方程 `i psi_t=-Delta psi/2+V psi`，状态为 `[Re psi, Im psi]`，使用 LDG 空间离散和 RK4。共享入口 `case/schrodinger.py` 提供五个算例：

| `--case` | 设置 | 默认终止时间 | 参考与输出指标 |
|---|---|---|---|
| `gaussian` | 自由高斯波包，中心 -4，动量 2，初始概率密度标准差 1 | 2 | 全空间解析波函数，复数 L2 误差 |
| `harmonic` | `V=x²/2`，初始中心 2、动量 0 的相干态 | `2 pi` | 解析波函数，一个完整周期 |
| `well` | `[0,1]` 无限深势阱，前两个归一化本征态等权叠加 | `4/(3 pi)` | 解析波函数，一个概率密度振荡周期 |
| `barrier` | `[0,2/3]` 内势高 6；高斯入射中心 -6、动量 3、标准差 .8 | 4 | 左区、势垒区、右区概率，以及总概率与能量变化 |
| `double-slit` | 二维有限高屏障 `0<=x<.75`，势高 30；两缝 `1<abs(y)<2` | 3 | 概率密度干涉图、三区概率及守恒指标 |

一维自由/谐振子/势垒计算域为 `[-16,16]`；二维域为 `[-12,12]×[-8,8]`，外边界均为齐次 Dirichlet。自由波包与谐振子的参考解属于全空间解，有限域截断误差未从 L2 误差中剔除；默认参数下边界尾部很小。势垒与双缝不使用吸收层，长时间计算会遇到外壁反射。双缝是**有限高势屏障模型**，不是带内部硬壁边界的几何网格；默认二维网格的单元面贴合屏障和缝的边缘，任意自定义 `--n` 不保证贴合。势垒默认一维网格也贴合其两边缘。

```bash
# 五例全部计算，生成 PNG、GIF、CSV、PT、守恒历史和 JSON
python -m case.schrodinger --case all
# 单例；默认势阱 n=16，其他一维 n=48，二维 n=32，p=3
python -m case.schrodinger --case harmonic --frames 120
python -m case.schrodinger --case double-slit --frames 120
# 解析解的三组网格收敛，最后一级 n=48
python -m case.schrodinger --case gaussian --n 12 --time .2 --convergence --no-plot
python -m case.schrodinger --case well --n 4 --convergence --no-plot
python -m unittest tests.test_second_order -v
```

输出位于 `results/schrodinger/<case>/`。`complex_l2_error` 是未消除整体相位的复波函数积分 L2 误差；质量为 `integral(|psi|²)`，能量为离散 `〈psi,H psi〉`。散射三区概率除以初始总概率，表示有限时间的概率分布，不能直接当作渐近反射/透射系数。未对初值或每一步重新归一化。

显式步长采用 `.5 / ((p+1)^4 sum(h_a^-2) + max|V|)`，仅针对本节默认模型设置的保守步长选择；不同本构模型需另行分析。默认 p=3 不意味着强形式离散 Laplacian 的点值残差也是四阶，应以演化解的网格收敛衡量整体精度。测试覆盖 1–3 维离散伴随性质、耗散恒等式、非零 Dirichlet 多项式精确性、扩散演化、复数周期模态、三例解析解收敛以及五例概率的半离散守恒。

LDG 薛定谔离散的背景参考：Lu、Cai、Zhang，
[Conservative Local Discontinuous Galerkin Methods for Time Dependent Schrödinger Equation (2005)](https://www.math.ualberta.ca/ijnam/Volume-2-2005/No-1-05/2005-01-06.pdf)。本实现使用上述明确列出的交替迹和 RK4，不声称复现论文全部格式。

五例已按上述默认参数实际计算至默认终止时间，并生成各 40 帧的 PNG/GIF 及数据文件（`results/schrodinger/`）：

| 算例 | 复波函数积分 L2 误差 | 总概率变化 |
|---|---:|---:|
| 自由波包，n=48 | 1.85963e-3 | -1.23853e-7 |
| 谐振子，n=48 | 1.70447e-4 | -1.22446e-9 |
| 无限深势阱，n=16 | 7.07710e-5 | -8.44391e-12 |
| 有限势垒，n=48 | 无解析瞬态参考 | -2.92162e-6 |
| 二维双缝，32×32 | 无解析瞬态参考 | -7.60397e-8 |

势阱 n=4/8/16 的误差为 `6.15542e-3 / 7.47806e-4 / 7.07710e-5`，加密阶为 3.041、3.401；这些实测结果不构成所有模型均达到四阶的声明。

#### 双缝条纹、对比度和透射概率对照

`case/double_slit_accuracy.py` 独立验证同一个有限高双缝势屏障、有限矩形域和齐次 Dirichlet 边界，不使用远场双缝公式代替本算例参考解。主方程解出复波函数；概率密度云图绘制 `|psi|²=Re(psi)²+Im(psi)²`，其区域积分才是区域内的概率。

```bash
python -m case.double_slit_accuracy --time-check \
  --production-solution results/schrodinger/double-slit/solution.pt
```

默认比较 p=3 的 32/64/128 单元每方向 FR-LDG 网格，与独立的 256/512/1024 点每方向二阶中心有限差分参考。全部分辨率均贴合势屏障和缝边缘；参考的边界 ghost 值取负内部值，以保持同样的零 Dirichlet 边界。两种离散的动能矩阵预先对角化，使用四阶对称分裂推进，FR 步长默认 `.00025`，参考步长默认 `.0005`；`--time-check` 对最细 FR 和最细参考再将各自步长减半，默认使用 2 个 CPU 线程。这个加速验证入口不修改原算例的 RK4 求解器；可选 `--production-solution` 会直接读取原 RK4 的输出进行比较。分裂格式的概率守恒不能代替时间精度检查。

固定终止时间 t=3，定义：

- 透射区域概率：`integral(x>=.75, |psi|²)/initial_probability`，是有限时间区域概率，不是渐近透射系数。
- 条纹曲线：`I(y)=integral(4.5<=x<6, |psi(x,y,3)|² dx)/initial_probability`，在 y∈[-5,5] 上取 2001 个共同探测点。对复振幅作局部插值后再求模平方。
- 中央峰和左右第一侧峰：在固定搜索窗内提取极大值，作三点抛物线位置细化；保留左右差异，不强制对称化。
- 对比度：左右分别用第一侧峰和它与中央峰之间的暗纹，计算 `(Imax-Imin)/(Imax+Imin)`。

结果位于 `results/double_slit_accuracy/`：`report.json`、`comparison.csv`、`profiles.csv`、条纹对照图 `fringes.png` 和单独的概率密度云图 `*_density.png`（青色屏障、白色探测带）。参考解自身的最后一次网格加密差异与时间步减半差异分别记录，比较误差不是对精确解误差的严格界限。`solution.pt` 缓存按方法、分辨率、时间和步长核对后复用。

本次双缝实算对照（t=3；探测带和对比度定义同上）：

| 方法 | 透射区域概率 | 中央峰 y | 右侧第一亮纹 y | 右侧对比度 |
|---|---:|---:|---:|---:|
| 原 RK4，FR 32×32，p=3 | 0.382782314 | -0.006682 | 2.587460 | 0.877339 |
| FR 64×64，分裂 dt=.00025 | 0.382630065 | -0.004112 | 2.577756 | 0.872870 |
| FR 128×128，分裂 dt=.000125 | 0.382640861 | 0.000280 | 2.580390 | 0.872843 |
| 独立 FD 1024×1024，dt=.00025 | 0.382592587 | 约 0 | 2.575850 | 0.872547 |

默认原 RK4 与细参考的透射概率相对差为 **0.04959%**，右侧峰位置差为 **0.01161**，对比度差为 **0.004791**，探测带曲线相对 L2 差为 **0.9122%**。同一 32×32 空间网格上，小步长分裂与原 RK4 的透射概率仅差约 `1.2e-8`。

但参考自身的 512→1024 网格变化仍为：透射概率 `1.579e-4`、右峰位置 `1.055e-3`、对比度 `1.216e-3`；参考时间步 .0005→.00025 的变化分别为 `8.261e-6 / 1.345e-6 / 7.472e-6`。因此上述差异不能当作对真解的精确误差，尤其不能宣称侧峰位置已经高精度收敛：细 FR 和细参考的峰位置仍存在差异，且加密变化不是一致单调的。概率守恒不能消除这些空间误差。

单独的默认 RK4 概率密度云图为 `results/schrodinger/double-slit/probability_density.png`；本次完整表格与图见 `results/double_slit_accuracy/summary.md`。
