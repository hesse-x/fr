# 二维等熵涡平移

入口：[isentropic_vortex.py](../isentropic_vortex.py)。以下命令均从仓库根目录执行。

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
