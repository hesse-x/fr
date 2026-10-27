# 固定网格背景流与现有 PML

入口：[acoustic_mean_flow_pml.py](../acoustic_mean_flow_pml.py)。以下命令均从仓库根目录执行。

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
