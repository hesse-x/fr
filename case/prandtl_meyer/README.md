# 普朗特–迈耶膨胀波

入口：[prandtl_meyer.py](../prandtl_meyer.py)。以下命令均从仓库根目录执行。

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
