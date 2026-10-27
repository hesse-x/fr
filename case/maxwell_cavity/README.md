# 三维 PEC 腔体电磁驻波

入口：[maxwell_cavity.py](../maxwell_cavity.py)。以下命令均从仓库根目录执行。

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

## 中心 / Roe 面通量与 PEC 壁面

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

## 解析模态与诊断

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

## 验证结果

CPU float64，`p=3,t=0.1,CFL=0.15,SSPRK(4,3)`：

| 网格 | 中心通量总场 L2 | 加密阶 | Roe 总场 L2 | 加密阶 |
|---|---:|---:|---:|---:|
| 2³ | 1.13417e-3 | — | 6.07434e-4 | — |
| 4³ | 1.60277e-4 | 2.823 | 4.77759e-5 | 3.668 |
| 8³ | 1.94160e-5 | 3.045 | 3.17499e-6 | 3.911 |

此设置中，三次多项式的中心通量场误差约三阶，Roe 约四阶；不能直接将之前 Euler/Rusanov 的四阶结果套到中心 Maxwell 通量。两种通量的误差均随加密下降。

`4³` 网格使用中心通量计算完整一个周期：总场 L2 误差 `1.16314e-4`，相对能量变化 `-2.56846e-7`，模态投影测得的相对频率误差 `2.89099e-7`；半离散能量变化率约 `-3.82e-19`。初始能量与解析能量均为 `0.138125`。该次实算结果位于 `results/maxwell_cavity/period/`。
