# 充分发展槽道 RANS

算例求解不可压、压力驱动、充分发展的定常槽道，采用 [Wilcox2006-LRN 低雷诺数 k–ω 模型](https://tmbwg.github.io/turbmodels/wilcox.html)。默认摩擦雷诺数为 `Reτ=178.12`，与 [MKM 槽道 DNS 参考数据](../channel-dns/reference/README.md) 一致。默认读取 [槽道 DNS 网格](../channel-dns/mesh/default.json)，其区域为 `[0,4πh] × [-h,h] × [0,4πh/3]`，半槽高为 `h=1`。流向和展向周期，上下壁面静止、无滑移。

## DNS、LES 与 RANS 对比

三组计算使用同一几何，并都与 `Reτ=178.12` 的不可压 MKM DNS 对照。本仓库的 DNS 和 LES 以恒质量流量驱动三维可压缩流动，`Ma=0.3`，从时均场确定实际 `Reτ`。RANS 给定压力梯度和 `Reτ`，体积平均速度由定常计算得到。

| 算例 | 湍流处理 | 网格与统计 | `Reτ` |
| --- | --- | --- | ---: |
| [本仓库 DNS](../channel-dns/README.md) | 无显式 SGS | `64×24×32, p=3`，第 60,000–100,000 步时均 | 下壁 182.46，上壁 182.65 |
| [本仓库 LES](../channel-les/README.md) | WALE，无壁面模型 | `32×12×16, p=3`，第 200,000–250,000 步时均 | 下壁 178.11，上壁 177.67 |
| 默认 RANS | Wilcox2006-LRN `k–ω` | 半槽 48 个壁法向控制体，直接求定常解 | 178.12（给定） |
| [MKM DNS 参考](../channel-dns/reference/README.md) | 不可压 DNS | 公开剖面 | 178.12 |

下表将各剖面线性插值到相同 `y⁺`。DNS 和 LES 的两项分别为下壁、上壁，按各自壁面尺度归一化。

| `y⁺` | 本仓库 DNS `U⁺` | LES `U⁺` | RANS `U⁺` | MKM DNS `U⁺` |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 8.500 / 8.561 | 8.637 / 8.649 | 8.451 | 8.522 |
| 30 | 13.617 / 13.761 | 14.048 / 14.160 | 12.392 | 13.868 |
| 100 | 16.741 / 16.775 | 17.617 / 17.749 | 16.322 | 17.147 |

- **近壁与外层速度。** 在 `y⁺=10`，三种方法均接近 MKM。在 `y⁺=30`，RANS 低约 10.6%。在 `y⁺=100`，本仓库 DNS 低约 2%–2.4%，LES 高约 2.7%–3.5%，RANS 低约 4.8%。[DNS 壁面单位数据](../channel-dns/results/mkm180-small-ma03-100k-20261027/postprocess/wall_units.csv) · [LES 平均速度图](../channel-les/figures/y_plus_u_plus_re180_p3_200k_250k.png)
- **壁面摩擦口径。** DNS 使用速度多项式端点导数，LES 表中使用第一近壁 Gauss 点的线性估算。将 LES 的同一时均场改用多项式端点导数，`Reτ` 为 190.44 / 189.43。应在统一剪切估算口径后比较摩擦，并考虑可压缩性、实际 `Reτ` 和统计区间的差异。[LES 壁面尺度计算](../channel-les/README.md#壁面尺度的计算口径)
- **模型标定。** RANS 在匹配 MKM 的 `Reτ` 时低估中心速度和体积平均速度约 4.7% 和 5.8%。壁法向加密后两者只变约 0.12%。拟合速度和剪应力可减小标定工况的平均速度误差，但两种 `Reτ` 下的湍动能误差均扩大。[网格对照](#rans-网格对照) · [低雷诺数阻尼标定](#低雷诺数阻尼标定)

## 定常计算

平均量只随离壁距离 `d` 变化。计算取从下壁到槽中心的半槽，使用 DNS 网格的 12 层壁法向单元及其 `p=3` Gauss 点，共 48 个未知位置，最近的解点为 `y⁺=0.492`。在这些点上以相邻点的中点划分控制体，对湍流输运方程做保守有限体积离散。

取半槽高和摩擦速度为单位，平均动量方程给出 `(ν+νt) dU⁺/d(d/h) = 1-d/h`，其中 `ν=1/Reτ`，驱动力对应的无量纲壁面剪应力为 1。用该关系消去平均速度梯度，再用 SciPy 的带状非线性求根直接求解 `k` 和 `ω` 的定常残差。模型保留低雷诺数修正、涡黏度应力限制和 `ω` 方程的交叉扩散项。壁面取 `k=0`，中心采用对称边界。对 `ω` 减去平滑壁面的 `6ν/(β₀(d/h)²)` 渐近项，离散有限的剩余量。

## 运行与结果

从仓库根目录运行。依赖已有的 PyTorch、NumPy、SciPy 和 Matplotlib。

```bash
venv/bin/python -m case.channel-rans
```

结果写到 `results/channel-rans/wilcox2006_lrn/`。可通过 `--output` 指定目录，`--re-tau` 和 `--mesh` 分别调整摩擦雷诺数和网格。自定义壁法向网格须关于中心对称，并在中心有单元面。

- `profiles.csv` 给出由壁至槽中心的半槽内部解点上的 `U⁺`、`k⁺=k/uτ²`、`ω⁺=ων/uτ²`、`νt/ν` 以及黏性与雷诺剪应力。
- `comparison_vs_mkm.csv` 和 `comparison_vs_mkm.png` 给出与 MKM DNS 的剖面对比。
- `summary.json` 记录残差、摩擦雷诺数、中心与体积平均速度、阻力系数和参考误差。

### RANS 网格对照

| `Reτ=178.12` | 中心速度 `Uc⁺` | 体积平均速度 `Ub⁺` |
| --- | ---: | ---: |
| RANS，默认 DNS 网格 | 17.4432 | 14.7782 |
| RANS，DNS 网格的壁法向各层二等分 | 17.4224 | 14.7598 |
| MKM DNS | 18.3010 | 15.6803 |

默认网格的最大缩放残差为 `1.02e-10`，平均速度在 MKM 测点上的均方根差为 `0.903 U⁺`。将壁法向各层二等分后，`Uc⁺` 只变化约 0.12%，在这两套网格上，离散敏感度比相对 DNS 的速度偏差小。

## 低雷诺数阻尼标定

用 MKM 的 `Reτ=178.12` 数据拟合 `R_k`、`R_β`、`R_ω`，保持模型的高雷诺数渐近系数和 `p=3` Gauss 解点。拟合在 24 层壁法向网格上进行，用 `Reτ=392.24` 的 MKM 数据在 48 层网格上独立检查。同一工况的标准与拟合系数用同一网格比较，并分别将网格二等分复核。拟合和评估使用在 `log(1+y⁺)` 上等距的 256 个位置。拟合目标中，各剖面的残差分别以标准设置在标定工况的 RMSE 归一化，系数搜索范围为各自标准值的 0.25 至 4 倍。

| 设置 | `(R_k, R_β, R_ω)` | `Reτ=178.12` 的 `U⁺ / −uv⁺ / k⁺` RMSE | `Reτ=392.24` 的 `U⁺ / −uv⁺ / k⁺` RMSE |
| --- | --- | --- | --- |
| 标准 | (6.000, 8.000, 2.610) | 0.778 / 0.0512 / 0.160 | 0.449 / 0.0432 / 0.179 |
| 拟合速度和剪应力 | (3.781, 2.000, 3.973) | 0.160 / 0.0263 / 0.621 | 0.405 / 0.0274 / 0.737 |
| 同时拟合湍动能 | (5.059, 6.971, 0.882) | 0.397 / 0.0398 / 0.163 | 0.653 / 0.0353 / 0.237 |

两项拟合将标定工况 `y⁺≈25.34` 的速度偏差从 −10.8% 降到 −2.0%，`R_β` 达到搜索下限，两个工况的湍动能误差均扩大。将湍动能纳入目标后，标定工况的局部速度偏差为 −6.8%，验证工况的速度 RMSE 比标准系数更大。算例默认采用标准系数。

从 [MKM 数据库](https://turbulence.oden.utexas.edu/MKM_1999.html) 下载原始参考表并运行两项标定。

```bash
ref_dir=results/channel-rans/calibration/reference
mkdir -p "$ref_dir"
for profile in chan180/profiles/chan180.reystress chan395/profiles/chan395.means chan395/profiles/chan395.reystress; do
  curl -fL "https://turbulence.oden.utexas.edu/data/MKM/$profile" -o "$ref_dir/${profile##*/}"
done
venv/bin/python -m case.channel-rans.calibrate
venv/bin/python -m case.channel-rans.calibrate --include-k
```

结果分别写到 `results/channel-rans/calibration/` 和其 `with_k/` 子目录。两组均输出对比图、拟合和验证工况的剖面表，以及包含加密网格结果的 `summary.json`。

## 可选近壁加密网格

[可选网格](mesh/default.json) 沿用 DNS 计算域，流向和展向等距，壁法向采用强度为 2.0 的对称 tanh 拉伸。表中的 `y⁺` 按参考 `Reτ=180` 换算。

| 网格量 | 数值 |
| --- | ---: |
| 流向 × 壁法向 × 展向单元数 | 4 × 32 × 4 |
| 总单元数 | 512 |
| 流向间距 `Δx/h` | 3.14159 |
| 展向间距 `Δz/h` | 1.04720 |
| 首层壁法向厚度 `Δy/h` | 0.0103548 |
| 槽中央单元的壁法向厚度 `Δy/h` | 0.128993 |
| 首层单元中心的参考 `y⁺` | 0.932 |
| `p=3` 最近 Gauss 解点的参考 `y⁺` | 0.129 |

网格采用 `fr-cartesian-v1` 格式。用以下命令从仓库根目录重新生成并运行这套网格。

```bash
venv/bin/python -m case.channel-les.generate_mesh \
  --cells 4 32 4 --stretch 2.0 --output case/channel-rans/mesh/default.json
venv/bin/python -m case.channel-rans --mesh case/channel-rans/mesh/default.json \
  --output results/channel-rans/wilcox2006_lrn_stretched
```
