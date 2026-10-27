# 二维柱面声脉冲与随流移动 PML

入口：[acoustic_pml.py](../acoustic_pml.py)。以下命令均从仓库根目录执行。

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

## PML 剖面与层厚对照

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
