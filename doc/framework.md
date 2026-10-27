# FR 框架与接口说明

算例路径索引见[根README](../README.md)，具体配置与结果见各算例README。

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
pip install -r requirements.txt

# 查看入口参数并运行一个算例
python -m case.ringleb --help
python -m case.ringleb --n 8 --p 3 --time .1

# 完整回归测试
python -m unittest discover -s tests -v
```

`requirements.txt` 统一包含数值计算、绘图和声学后处理依赖；运行时可用相应入口的 `--no-plot` 关闭绘图。Gmsh 网格生成另外需要 PATH 中的 `gmsh` 可执行文件，读入已有网格无需调用 Gmsh。

`--n` 通常表示每个方向的单元数，`--p` 为多项式次数；不同入口的参数以 `--help` 为准。Cartesian 每单元有 `(p+1)^d` 个解点。部分入口提供 `--threads`、`--device`、`--cfl`；默认的小网格计算通常使用 CPU 单线程。

### 适用范围与待支持功能

- Euler 尚无激波限制器、正性保持器、非线性去混叠或 Roe 通量；非物理密度/压力会报错。
- 二维非结构离散实现 DG/FR-DG 成员，尚无完整 VCJH 参数族、非共形面或三维非结构体网格。
- 二阶 LDG 尚未扩展到曲边非结构网格，也未实现 Neumann/混合条件。通用方程接口未内置完整NS；槽道及翼型NS算例在各自目录组装黏性应力和壁面条件。
- 固定网格斜向背景流直接复用 PML 属于数值实验，其一般稳定性和长时间完美匹配尚未证明。

## 模块分工与计算流程

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

目前通用空间离散支持 1–3 维 Cartesian 张量积网格，每个方向可设置不同单元数和区域范围，默认单个方向等距；可通过 `edges` 指定非均匀单元边界。所有方向使用相同多项式次数。曲边和非结构二维网格使用独立的 `MeshFR2D` 接口，见下文。

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

背景流PML算例的具体说明见[算例README](../case/acoustic_mean_flow_pml/README.md)。例如选择完整线化Euler：

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

依据 [Castonguay–Vincent–Jameson 的三角形 FR 论文](<A New Class of High-Order Energy Stable Flux Reconstruction Schemes for Triangular Elements.pdf>) 第 5.4 节，首先采用 VCJH 的 **c=0，即 DG 成员**。三角形使用完整 `P_p` 多项式空间，四边形使用张量积 `Q_p`。`p=3` 时每个三角形为 10 个自由度，每个四边形为 16 个自由度；面上积分点数另行确定。

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

### 验证组织

`tests/` 覆盖参考单元恒等式、多项式精确性、自由流保持、面通量平衡、非自治时间推进、解析解收敛、边界条件、PML 参考对照、二阶算子伴随性质与双缝独立参考。最近一次通用回归共 **83 项测试通过**，槽道专用回归另有 **23 项测试通过**。以下算例中的误差表是各自列明参数下的实测结果。

