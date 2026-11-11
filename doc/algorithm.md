# 方程离散与边界条件实现

本文按物理领域说明仓库中**已实现**的方程、空间离散、界面通量、边界条件与时间推进。
重点是各类算例共同使用的算法；具体物理参数、运行命令和结果见各算例README。
公共接口的完整参数说明见 [framework.md](framework.md)，算例入口见[根README](../README.md)。

## 1. 总体组织

多数算例的离散过程为：

```text
守恒量/场变量
  → 单元内多项式表示
  → 体通量与面上左右状态
  → 内部面公共通量 / 边界外部状态
  → FR修正或DG面提升
  → 半离散RHS
  → RK或专用时间推进
  → 空间积分、载荷、相量、能谱等诊断
```

方程负责物理通量及波速，空间算子负责几何、插值、积分和面连接；
边界由回调、预配置或算例的原生适配层提供。二阶项还需要独立指定梯度迹和黏性通量，
不能只给一阶Riemann求解器配置一个外部状态。

| 领域 | 状态与模型 | 主要空间路径 | 边界处理 |
|---|---|---|---|
| 标量输运 | 标量平流 | Cartesian FR / 二维曲边FR-DG | 周期连接、解析给定外部状态 |
| 可压缩无黏流 | 非线性Euler守恒量 | Cartesian FR / 二维曲边FR-DG | 周期、解析状态、滑移壁、特征远场 |
| 可压缩黏性流 | 理想气体NS，常黏度 | Cartesian / 二维曲边均可选LDG或BR1加惩罚 | 周期、恒温无滑移壁、绝热无滑移壁、远场 |
| 声学与线化流动 | 压力/速度扰动；线化Euler另含密度扰动 | Cartesian FR，特征通量，方向分裂PML | 刚壁、PML外端条件；背景流需区分坐标处理 |
| 电磁波 | Maxwell六分量；二维TMz三分量 | Cartesian FR / 二维曲边FR-DG | PEC、材料阻抗界面、PML、入射特征为零的辐射外边界 |
| 量子波动 | 薛定谔实虚部或复振幅 | Cartesian FR-LDG Hamiltonian | 外边界零Dirichlet、体势垒/势阱 |

原生实现是上述若干离散路径的执行后端，支持范围由各自入口检查。
它们没有共同的、可以覆盖所有方程和任意边界回调的通用C++求解器。

## 2. 公共空间离散、通量与时间推进

### 2.1 Cartesian FR-DG

[fr/core.py](../fr/core.py) 支持1～3维张量积Cartesian网格。
每方向使用p+1个Legendre–Gauss解点，以Lagrange多项式表示单元内状态；
微分矩阵、左右端点插值向量和Radau修正导数由 [fr/operators.py](../fr/operators.py) 构造。

对一维映射单元，守恒方程的半离散形式为：

$$
 \frac{dU}{dt}=-\frac{2}{h}\left[
 DF+g'_L(\widehat F_L-F_L^D)+g'_R(\widehat F_R-F_R^D)
 \right]+S.
$$

多维按方向分别处理后相加。这里：

- 体通量在解点或额外积分点求值。
- 面状态由状态多项式插值得到，再计算公共数值通量。
- 修正项减去的是体离散通量的端点插值，保留非线性通量的离散守恒关系。
- 可通过`edges`提供非均匀单元边界，各方向微分及CFL使用实际局部宽度。
- `anti_alias='flux'`或`'surf-flux'`分别对体/面通量增加Gauss积分并作L₂投影；
  解点数不变，源项不自动去混叠。

这条路径实现FR-DG成员。当前代码没有完整的可调VCJH参数族、限制器或正性保持器。

### 2.2 二维曲边FR-DG

[fr/unstructured.py](../fr/unstructured.py) 的`MeshFR2D`支持共形三角形、四边形及混合网格。
[fr/reference.py](../fr/reference.py) 提供参考基与积分规则，
[fr/mesh.py](../fr/mesh.py) 和 [fr/gmsh.py](../fr/gmsh.py) 提供网格及Gmsh读取。

算法使用几何映射、物理质量矩阵、体积分矩阵和面提升：

$$
 M\dot U=G_xF_x+G_yF_y-\mathcal L(\widehat F_n)+M S.
$$

通量在额外体积分点求值，面上使用法向通量；每类单元独立成块，预计算几何矩阵，
通过Cholesky求解施加质量矩阵逆。相邻面的积分点方向在连接时对齐。
构造时检查Jacobian及面长度，拒绝翻转、折叠或退化单元。
这条路径对应DG/三角形VCJH的c=0成员，尚无三维曲边非结构网格或非共形面实现。

### 2.3 面通量与边界接口

| 接口 | 输入 | 返回值/约定 |
|---|---|---|
| Cartesian `flux(U,axis)` | 状态及坐标轴 | 正坐标方向的物理通量 |
| Cartesian `numerical_flux(minus,plus,axis)` | 两侧`Face`，含状态及物理通量 | 正坐标方向公共通量；minus始终为坐标较小侧 |
| Cartesian `boundary(coords,inside,axis,side,t)` | 域内面状态，side=−1或+1 | **外部状态**；不是直接覆盖解点 |
| 曲边 `numerical_flux(minus,plus,normal)` | 两侧`NormalFace`及单位法向 | 当前单元外法向公共通量，`Face.flux`已包含法向点乘 |
| 曲边 `boundary(coords,inside,normal,tag,t)` | 面积分点、域内状态、法向和网格标签 | 外部状态 |
| LDG `boundary(coords,inside,axis,side,t)` | 待求梯度变量的域内迹 | **边界值**，与上述外部状态接口含义不同 |

周期方向直接连接首尾面，不调用物理边界回调。
公共边界函数与按面/标签组合器见 [fr/boundaries.py](../fr/boundaries.py)。

[fr/fluxes.py](../fr/fluxes.py) 提供：

- 中心通量：两侧物理通量平均。
- Rusanov：中心通量减去最大局部波速乘状态跳跃的一半。
- 特征Roe/upwind：由方程提供`|A|`或`|A_n|`作用；Cartesian Euler/NS、线性声学、线化Euler和Maxwell提供相应接口。
- 标量迎风：按常速度符号选择上游通量。
- Richtmyer/Lax–Wendroff面预测：以实际步长预测界面状态，再送入FR/RK；需调用`prepare_step`。

Cartesian Euler/NS 的 Roe 采用 Roe 平均，声学波使用宽度为 0.05c 的 Harten 二次熵修正；
接触/剪切波保留 |u_n|。它不包含低 Mach 预处理。不规则网格 Euler/NS 的一般法向 Roe 尚未实现。

### 2.4 时间推进

[fr/time.py](../fr/time.py) 提供SSPRK(4,3)、经典RK4及自定义步进接口，
每个子阶段以相应阶段时间评估RHS；到结束时刻或采样端点时截短实际步长。

Cartesian双曲步长按方向波速与单元宽度估计：

$$
 \Delta t\le\frac{\mathrm{CFL}}
 {(2p+1)\max_{\mathbf x}\sum_a \lambda_a/h_a}.
$$

显式扩散/薛定谔还需考虑约为`(p+1)^4/h²`的谱尺度；PML还受阻尼率限制。
具体系数由对应模型或算例配置，不能将一个CFL值跨离散路径直接等同。

翼型原生求解器允许`global`物理时间或`local`局部伪时间。
局部伪时间用于定常残差收敛，不代表真实非定常流动时间；
圆柱脱涡、槽道和Taylor–Green统计使用全局物理时间。

## 3. 标量平流与非线性Euler

### 3.1 平流

方程为`u_t + ∇·(a u)=0`。
Cartesian平流采用常速度迎风通量和周期面连接；二维曲边平流采用法向Rusanov，
其波速为`|a·n|`，常速度下等价于迎风。
外边界可以使用解析随时间变化的外部状态，以检查曲边几何和传播误差。

对应组装示例：[advection](../case/advection/solve.py)、[mesh_advection](../case/mesh_advection/solve.py)。

### 3.2 Euler方程与边界

[fr/equations/euler.py](../fr/equations/euler.py) 使用：

$$
 U=(\rho,\rho\mathbf u,E),\qquad
 p=(\gamma-1)\left(E-\tfrac12\rho|\mathbf u|^2\right),
$$

$$
 F_a=(\rho u_a,\rho\mathbf u u_a+p\mathbf e_a,(E+p)u_a).
$$

Rusanov波速为`|u_a|+c`或`|u·n|+c`。
有效性检查要求状态有限、ρ>0、p>0；检查会报错，不会自动修复非物理状态。

边界处理有三种主要组装方式：

1. **周期连接**：用于等熵涡对流，守恒量跨周期面交换。
2. **解析给定状态**：Ringleb和Prandtl–Meyer的边界由解析场生成；
   当前是有限时间离散误差检查，不以精确残差抵消构造离散解。
3. **滑移壁与远场**：二维曲边翼型使用法向动量反射，保持ρ、E；
   远场组合出射/入射Riemann不变量，按流向选择熵及切向速度。

翼型的壁面公共通量进一步写成质量/能量通量为零、动量通量为`p_wall n`。
`p_wall`保留反射状态Rusanov通量中的法向声学阻抗项，不能直接改成未修正的域内静压。
该特征远场实现固定γ=1.4、ρ∞=c∞=1，属于专门的归一化适配。
实现见 [翼型Euler组装](../case/naca0012-euler/airfoil.py) 和
[原生公共边界](../fr/native/boundaries.hpp)。

## 4. Navier–Stokes：公共本构与可配置黏性离散

### 4.1 常物性本构

[fr/equations/navier_stokes.py](../fr/equations/navier_stokes.py) 在Euler基础上提供常黏度应力和导热：

$$
 \tau=\mu\left(\nabla\mathbf u+\nabla\mathbf u^T-
 \tfrac23(\nabla\cdot\mathbf u)I\right),
 \qquad \kappa=\frac{\mu\gamma}{(\gamma-1)Pr}\quad(R=1),
$$

$$
 F^v_a=\left(0,\tau_{:a},\mathbf u\cdot\tau_{:a}+\kappa\partial_a T\right),
 \qquad \dot U=-\nabla\cdot F^c+\nabla\cdot F^v+S.
$$

三维Cartesian公共组装使用R=1、T=p/ρ。
二维曲边实现以θ=p/ρ=RT为传热变量，并配套相应导热系数。
两条路径都采用Stokes假设的`2/3`系数。

### 4.2 公共黏性接口与Cartesian实现

[fr/viscous.py](../fr/viscous.py) 的`make_viscous_operator`独立选择黏性离散方法，
并按网格选择对应实现。算法不是由维数或曲边属性强制决定的。

| 网格 / 后端 | `ldg` | `br1-penalty` |
|---|---|---|
| 1–3维Cartesian，Torch公共黏性算子 | 支持 | 支持 |
| 三维Cartesian NS，CPU FP64 p=3 native | 支持，原有默认 | 支持，保持组内融合 |
| 二维曲边三角形/四边形/混合网格，Torch | 支持 | 支持 |
| 二维曲边同类单元块，CPU FP64 native | 支持 | 支持，翼型/圆柱原有默认 |

两种实现提供相同的`gradient(W,t)`、`flux(W,gradient)`、`evaluate(U,t)`和`rhs(U,t)`接口。
W为速度和温度；`evaluate`返回黏性RHS及数值面通量，Cartesian的第二项为`None`。
网格几何、气体本构、物理边界回调与算法选择分别配置：

```python
from fr import make_viscous_operator

viscous = make_viscous_operator(
    grid, ns_equation, scheme='br1-penalty', penalty=4.,
    boundary=primitive_boundary,
)
viscous_rhs = viscous.rhs(U, t)
```

Cartesian边界回调签名为`(coords,inside,axis,side,t)`，曲边为`(coords,inside,normal,tag,t)`，
均返回实际原始变量边界值。曲边可另传`normal_flux(coords,inside,normal,tag,t,flux)`，
返回施加物理条件后的外法向黏性通量；例如静止绝热壁将能量通量置零。
Cartesian NS的恒温壁机械功修正由其公共组装执行。

[fr/navier_stokes.py](../fr/navier_stokes.py) 的`CartesianNavierStokes`组装FR/Rusanov对流和
所选黏性算子。默认[CartesianLDG](../fr/second_order.py)步骤如下：

1. 从守恒量得到`(u,v,w,T)`。
2. 梯度阶段内部面取坐标较小侧迹，即`w_hat=w_minus`。
3. 应用本构生成应力和总黏性通量。
4. 散度阶段取互补的较大侧通量迹，即`Fv_hat=Fv_plus`。

周期或齐次Dirichlet情况下，这对算子在Gauss质量内积中满足离散伴随关系`D=−G*`，
使标量扩散具有非正的能量贡献。耦合应力/导热复用同一梯度与散度接口。
`br1-penalty`改用平均梯度迹和平均黏性通量，并加入速度/温度跳跃惩罚，
系数分别为`μ C(p+1)²/h_face`和`κ C(p+1)²/h_face`，h_face取相邻单元较小宽度。
公共Cartesian二阶接口尚无通用Neumann/Robin配置。

公共NS边界预配置：

| 配置 | 对流处理 | 黏性处理 |
|---|---|---|
| `periodic` | 三向首尾连接 | 梯度/散度均周期连接 |
| `isothermal-channel` | x/z周期；y壁反射法向动量 | 壁面速度迹为零、温度迹为Tw；修正静止壁面的机械功通量 |

静止壁面不做机械功，但恒温壁仍允许导热。
对流反射本身只约束法向运动；无滑移的切向速度由黏性梯度迹施加。
边界条件在数值迹/通量中实现，不要求最靠壁面的内部Gauss解点速度强制置零。

均匀x向体积力G的源项为`S_(rho u)=G`、`S_E=G u`。
`constant_mass_flux=True`由每个RK阶段的无驱动力动量RHS体积分求G，维持体积平均ρu；
不逐点缩放速度。参考Python实现见 [fr/sources.py](../fr/sources.py)。
该约束显式启用，Taylor–Green的三向周期无驱动预配置不启用它。

### 4.3 `native-p3`执行预配置

[fr/native/cartesian_ns/](../fr/native/cartesian_ns/) 支持三维Cartesian、CPU FP64、p=3、SoA状态，
以及上述周期/恒温槽道两种边界。
`make_stepper('native-p3')`使用4×4×4单元分组、线程私有halo/scratch、AVX2/FMA，
组内融合对流、所选黏性算子和局部RK更新，并将有效性/CFL检查合入末阶段。
可选恒流量约束和12场矩梯形累计使用独立显式接口。

`viscous_scheme='ldg'`或`'br1-penalty'`在模型构造时选择。
native通过[viscous_kernel.hpp](../fr/native/cartesian_ns/kernels/viscous_kernel.hpp)编译期选择
[LDG](../fr/native/cartesian_ns/kernels/ldg_kernel.hpp)或
[BR1加惩罚](../fr/native/cartesian_ns/kernels/br1_kernel.hpp)，
不把黏性计算拆成额外的全域kernel。LDG保留原有单侧依赖裁剪；BR1需要两侧halo，
因此不能假设两者开销相同。BR1惩罚系数进入构建缓存键。

边界规则位于 [kernels/boundaries.hpp](../fr/native/cartesian_ns/kernels/boundaries.hpp)。
默认`boundary_mode='separate'`在组内分别遍历内部与边界面/单元，继续复用scratch；
它没有引入全域边界数组或额外全域同步。`'fused'`保留作对照。
native-p3 与 Torch 均支持 `interface_flux='rusanov'|'roe'`，以及
`anti_alias='none'|'flux'|'surf-flux'|'flux, surf-flux'`；默认仍为 Rusanov、无去混叠。
原生界面 Roe 使用 [roe.hpp](../fr/native/cartesian_ns/kernels/roe.hpp)，
体/面 Gauss 过积分与 L₂ 投影使用 [dealiasing.hpp](../fr/native/cartesian_ns/kernels/dealiasing.hpp)，
复用公共 FR 插值/投影矩阵并保持解自由度不变。模式进入编译缓存键，默认分支不计算额外求积。
任意阶或 GPU 应使用 Torch 参考路径；原生入口会拒绝不支持的配置。

槽道及Taylor–Green共享这条NS路径；槽道的拉伸网格、初场、扰动、统计窗口和检查点调度留在算例。
相应数据流与性能记录见 [channel-dns/algorithm.md](../case/channel-dns/algorithm.md) 和
[channel-dns/perf.md](../case/channel-dns/perf.md)。

### 4.4 二维曲边：平均迹或互补迹

曲边黏性几何、梯度提升、通量和惩罚位于[fr/viscous.py](../fr/viscous.py)，
原生算子位于 [fr/native/compressible.cpp](../fr/native/compressible.cpp)。
[翼型适配器](../case/naca0012-ns/airfoil.py)只提供归一化参数、边界值和载荷后处理。

默认`br1-penalty`先插值速度和θ，在内部面取两侧平均迹构造提升梯度；
再计算体黏性通量，面上取两侧黏性通量平均并加入速度/温度跳跃惩罚。
惩罚尺度与`μ(p+1)²/h_face`相关，面尺寸取相邻单元的约束尺度。
这是BR1加显式跳跃惩罚，不称为标准BR2。

选择`ldg`时，一对内部面以较小面编号侧提供原始变量迹，较大面编号侧提供黏性通量，
不另加内部跳跃惩罚。共享面的积分点反向由邻接映射处理；几何变换、质量矩阵和面法向
仍由曲边网格负责。曲边无需保角或正交，LDG也不局限于Cartesian网格。

- **静止绝热无滑移壁**：速度迹为零、θ取内侧迹，温度惩罚跳跃为零；
  面上的总黏性能量通量显式置零。
- **远场**：梯度迹及黏性跳跃使用给定自由流原始变量；对流仍使用特征远场。
- **归一化**：ρ∞=c∞=参考长度=1，u∞=Ma，因此μ=Ma/Re，
  与Cartesian Taylor–Green中U₀=1、μ=1/Re的参数表示不同。

[fr/native/compressible.py](../fr/native/compressible.py) 复用预计算几何矩阵、工作区和显式RK；
翼型Euler、翼型NS和圆柱NS通过适配层选择黏性、边界标签和几何。
圆柱流复用翼型NS驱动并改为直径归一化。这一后端目前要求其适配器支持的同类二维单元块，
不能据此认为所有`MeshFR2D`混合网格都已有原生NS支持。

槽道/Taylor–Green保持默认LDG，翼型/圆柱保持默认BR1加惩罚。
四个NS入口提供`--viscous-scheme {ldg,br1-penalty}`和`--penalty`；
续算检查该配置，旧检查点缺少字段时按原算法和惩罚系数4解释。
步长估计考虑BR1惩罚系数；它是保守估计，并非所有网格上的严格稳定界。
新增路径已有解析梯度/黏性RHS、耗散、守恒、绝热壁和native/Torch多阶段一致性测试，
尚未用新选项重算已有完整DNS或气动力基准，原有结果仍对应原默认方法。

## 5. 声学、线化Euler与PML

### 5.1 两种线性方程

[fr/equations/acoustics.py](../fr/equations/acoustics.py) 使用`(p',u')`：

$$
 (\partial_t+\mathbf U_0\cdot\nabla)p'+\rho_0c_0^2\nabla\cdot\mathbf u'=0,
\qquad
 (\partial_t+\mathbf U_0\cdot\nabla)\mathbf u'+\nabla p'/\rho_0=0.
$$

`Acoustics`取U₀=0，`ConvectedAcoustics`加入均匀背景平流。
[LinearizedEuler](../fr/equations/linearized_euler.py) 使用`(rho',u',p')`，额外推进密度连续性方程，
包含独立的平流熵模态`rho'−p'/c₀²`。
两者均提供特征绝对Jacobian作用，面耗散针对入射/出射声学模态；
背景流下横向速度和熵模态也以背景速度输运。
这些模型没有背景剪切、背景密度梯度或黏性项。

### 5.2 刚壁与方向分裂PML

静止声学刚壁将法向速度扰动取反，保持压力和切向速度。
它用于声学/相容背景条件，不等同于非线性NS的无滑移恒温壁。

[fr/pml.py](../fr/pml.py) 采用方向分裂：物理状态`U=Σ_a Q_a`，
第a个分量只接收该方向的通量散度，并附加`−σ_a Q_a`。
σ从内部区域的零值向层外侧按幂次平滑增大；其峰值由目标反射参数、波速和厚度估算。
PML是有体积厚度的吸收区域，层最外端仍需边界条件。

两种背景流使用方式应分别理解：

| 使用方式 | 当前实现 |
|---|---|
| [acoustic_pml](../case/acoustic_pml/solve.py) | 在随均匀背景速度平移的坐标中求静止声学，物理坐标随时间换算，吸收层随坐标系移动 |
| [acoustic_mean_flow_pml](../case/acoustic_mean_flow_pml/solve.py) | 固定网格上对ConvectedAcoustics/LinearizedEuler应用现有SplitPML，属于数值实验 |

现有SplitPML没有针对任意斜向均匀背景流重新推导匹配条件，
固定背景流实验不宣称一般稳定性或长时间完美匹配。
验证使用扩大无吸收域的核心区误差；核心误差比和PML设计反射参数都不是实测入射/反射系数。

### 5.3 流动声学：FW-H曲面积分后处理

圆柱流还提供从非线性NS结果计算辐射声的路径。
[fwh_sampling.py](../case/cylinder_flow/fwh_sampling.py) 在全局物理时间推进中，
对静止闭合轮廓采样FR面的守恒数值质量通量与动量通量，并检查轮廓积分闭合性。
[fwh.py](../case/cylinder_flow/fwh.py) 对周期信号作谐波拟合，
使用均匀亚声速背景流的二维频域Green函数计算FW-H曲面积分。

该方法是声学类比后处理，不在整个外域推进声学方程；
当前省略轮廓外的四极子体积项，因此保留不同可渗透轮廓的结果作敏感性检查。

## 6. Maxwell、材料界面与电磁开放域

### 6.1 均匀材料Maxwell

[fr/equations/maxwell.py](../fr/equations/maxwell.py) 的六分量状态为`(Ex,Ey,Ez,Hx,Hy,Hz)`：

$$
 \epsilon\partial_t\mathbf E=\nabla\times\mathbf H,\qquad
 \mu\partial_t\mathbf H=-\nabla\times\mathbf E.
$$

均匀材料波速为`c=1/sqrt(epsilon*mu)`。
特征通量只对切向场跳跃加耗散，法向零速模态不附加标量Rusanov式惩罚。
闭腔可用中心或特征上风通量，显式RK推进；能量漂移包含时间离散及上风耗散的影响。
模型没有额外的散度清理方程。

**PEC外部状态**：E切向反号、E法向不变；H法向反号、H切向不变，
使平均迹满足E_t=0、H_n=0。
二维TMz状态为`(Ez,Hx,Hy)`，有对应的Cartesian与任意面法向PEC处理。
公共Python与C++规则分别在 [fr/boundaries.py](../fr/boundaries.py) 和
[fr/native/boundaries.hpp](../fr/native/boundaries.hpp)。

### 6.2 介质界面TMz与散射

均匀六分量Maxwell类本身没有通用变材料界面组装。
[介电圆柱散射](../case/mie_cylinder/solve.py) 专门采用二维容量形式：

$$
 C\,\partial_t U+\nabla\cdot F_{\mathrm{curl}}=S,
 \qquad C=\operatorname{diag}(\epsilon,\mu,\mu).
$$

材料区域由网格region标签给定；界面根据两侧阻抗`Z=sqrt(mu/epsilon)`求公共电场和切向磁场，
再生成公共法向curl通量。曲边网格负责圆柱贴体几何。

这个算例推进散射场，已知入射波在体源中以`(I−C)∂t U_inc`进入方程；
PML源也按容量配套，空间RHS最终除以C。
仿真经过平滑启动和多个周期后，对接受的物理状态做相量积分，与圆柱级数解析解比较。

### 6.3 辐射、天线、源与几何

三维阵列与抛物面天线采用时域Maxwell、周期驱动和后处理相量，
并非直接求解频域Helmholtz方程。

- **PML外域**：复用方向分裂状态，三维六分量场展开为18个分裂变量，叠加方向阻尼。
- **Silver–Müller选项**：抛物面原生路径在外部迹置零后使用特征上风通量，
  实现入射特征为零并保留出射波的通量；不是把边界内部六分量全部强制清零。
- **抛物面实体**：由单元中心判定固体mask，PEC位于相邻Cartesian单元面，属于阶梯几何近似；
  固体单元不演化，表面几何精度受网格细化约束。
- **电流馈源**：平滑局部体电流，按空间积分归一化后施加平滑启动的周期驱动。
- **电压馈源**：离散电场线积分约束；原生RK4子阶段及最终状态均投影到对应时刻的目标电压。

[parabolic_antenna/native.cpp](../case/parabolic_antenna/native.cpp) 实现其场更新、PEC、辐射外边界和馈电投影；
这条生产路径具有专门的执行约束，不能只依据Python构造参数推断任意边界都已经原生支持。
[uniform_array](../case/uniform_array/solve.py) 按阵元位置和相位构造体源。
闭合采样面上的E/H相量进一步构成等效电磁流，用于近远场变换和方向图比较。
波形预热、每周期相量变化和采样面位置都是后处理有效性的检查项。

## 7. 薛定谔：公共LDG Hamiltonian与两种传播方式

### 7.1 一维实虚部与RK4

[fr/equations/schrodinger.py](../fr/equations/schrodinger.py) 使用hbar=m=1的方程：

$$
 i\partial_t\psi=-\tfrac12\Delta\psi+V\psi.
$$

以`U=(Re psi, Im psi)`存储，先用LDG构造`H(U)=−0.5 D G U+VU`，
再返回`(H_Im,−H_Re)`。
[一维算例](../case/schrodinger/solve.py) 包括自由波包、谐振子、无限深势阱和有限势垒，使用RK4。

外边界在LDG梯度阶段指定零Dirichlet值；散度采用互补迹。
Hamiltonian在相应离散内积中具有自伴结构，概率和能量用质量权积分检查。
RK4不是严格幺正推进，有限时间步仍可产生概率/能量漂移。
有限计算盒的零边界会反射抵达边界的波包；这里没有通用量子PML或吸收势配置。

### 7.2 二维双缝：FR-LDG与Chebyshev指数传播

[quantum_double_slit/solver.py](../case/quantum_double_slit/solver.py) 复用一维FR-LDG生成每个方向的稀疏动能矩阵，
按Gauss质量权平方根变换到Euclidean内积，组合为二维张量积Hamiltonian。
该算例的无量纲约定是`H=−Δ/(4π²)+V`，与上一节的`−Δ/2`系数不同。

- 外盒为零Dirichlet边界；双缝屏障由体内有限高势能区域表示，缝口为低势能区域。
- 原生传播在 [evolve.cpp](../case/quantum_double_slit/evolve.cpp) 中进行稀疏Hamiltonian作用和Chebyshev递推。
- 用谱上界缩放Hamiltonian，Bessel系数控制指数展开截断误差；
  `--dt`是指数传播分块长度，增大它会改变多项式阶数，不是显式RK的稳定时间步。
- 传播误差受展开截断及浮点运算影响，需检查概率、能量和晚期角分布稳定性。
- FFT用于透射波包的远场角分布后处理；实际空间推进仍使用FR-LDG矩阵。

此双缝实现为零外场、静态势能问题；没有实现一般电磁耦合的量子演化。

## 8. 边界类型的选用与验证

| 物理要求 | 当前处理位置 | 应检查的量 |
|---|---|---|
| 周期传播/周期黏性流 | Cartesian面连接；LDG互补周期迹 | 周期接缝、质量/动量/总能量、解析解误差 |
| Euler滑移壁 | 法向动量反射及壁面公共通量 | 法向质量通量、压力载荷、几何法向 |
| NS恒温无滑移壁 | Cartesian速度/温度Dirichlet迹及壁面功修正 | 壁面速度迹、壁温、剪切与导热 |
| NS绝热无滑移壁 | 曲边BR1速度迹、温度跳跃及能量通量修正 | 黏性牵引、壁面能量通量、边界层分辨率 |
| 声学刚壁 | 法向速度扰动反射 | 反射脉冲、声学能量及扩大域误差 |
| 电磁PEC | 电/磁场对应分量反射 | 切向E、法向H、腔模误差及能量 |
| 介质界面 | 材料阻抗Riemann状态及容量矩阵 | 切向场连续性、相量与解析散射 |
| 开放波动域 | PML体阻尼或特征辐射外边界 | 扩大域差异、预热收敛、外边界距离/层厚敏感性 |
| 量子外盒/势垒 | LDG零Dirichlet；体势能 | 概率、能量、透射/反射及边界反射到达时间 |

不同物理量的守恒预期不同：周期无驱动NS中动能因黏性衰减，但总能量仍应守恒；
PML区域的能量下降是设计行为；PEC加中心通量的半离散能量性质不代表显式RK严格守恒；
薛定谔概率守恒与能量守恒也需分别检查。

接口/算子验证可参考 [tests/test_modular.py](../tests/test_modular.py)、
[tests/test_unstructured.py](../tests/test_unstructured.py)、
[tests/test_second_order.py](../tests/test_second_order.py)、
[tests/test_equations.py](../tests/test_equations.py)。
物理模型与原生后端还各有对应算例测试。
Python/原生一致性、守恒和已有参考吻合是不同层次的证据，
不能由单项检查替代网格、时间步、边界位置或几何近似的敏感性验证。
