# 算例运行与输出约定

各算例路径和计算内容见[根README](../README.md)。参数、网格、命令及实测结果见各自README。

## 云图与动画

直接运行算例就会在 `--output` 目录中生成图片，原有 CSV/PT/VTK 等输出继续保留。

| 算例 | 最后时刻 PNG | 非定常 GIF |
|---|---|---|
| `ringleb`、`prandtl_meyer` | 密度、压力、Mach 三幅云图 | 定常问题，不生成动画 |
| `isentropic_vortex` | 密度、压力、Mach | 三个物理量随时间演化 |
| `acoustic_pml`、`acoustic_mean_flow_pml` | 压力扰动 | 柱面波传播及 PML 吸收 |
| `maxwell_cavity` | Ez、电场模、磁场模 | 同一物理截面的驻波振荡 |
| `uniform_array` | xy 截面 Ez、远场功率方向图及解析对照 | 阵列辐射的建立过程 |
| `advection` | 标量场；1D 为曲线 | 1D/2D/3D 对流演化 |
| `mesh_advection` | 曲边混合网格上的标量云图 | 标量传播 |
| `schrodinger` | 一维概率密度曲线/二维概率密度云图及波函数实部 | 波包传播、势阱振荡和双缝干涉 |
| `mie_cylinder` | 瞬时散射 Ez，以及采样所得散射 Ez 相量幅值 | 瞬时散射 Ez 从启动到结束的演化 |
| `cylinder_flow` | 二维圆柱绕流涡量 | 全局物理时间推进的脱涡过程，见[运行参数](cylinder_flow/README.md) |

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

通用非定常算例默认 `--frames 60 --fps 12`，`cylinder_flow` 默认 `--frames 120 --fps 15`；`--frames 0` 只生成最后一步 PNG，
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
