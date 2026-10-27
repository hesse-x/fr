# 含时薛定谔方程：五个经典算例

入口：[schrodinger.py](../schrodinger.py)。以下命令均从仓库根目录执行。

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


双缝独立精度验证见[double_slit_accuracy/README.md](../double_slit_accuracy/README.md)。
