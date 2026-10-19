# NACA 0012 二维 Euler 算例

独立的翼型网格生成和实验性 Euler 绕流计算入口。所有命令从仓库根目录执行，使用仓库的 `fr` 求解核心；本目录名带连字符，因此用脚本路径运行。

## 生成网格

依赖 PATH 中的 `gmsh`，以及仓库 Python 环境。Matplotlib 用于可选 PNG 预览。

```bash
source venv/bin/activate
python case/naca0012-euler/generate_mesh.py
```

默认输出到本目录的 `results/mesh/`：

- `naca0012.geo`：可修改、可用 Gmsh 打开的几何与尺寸场。
- `naca0012.msh`：ASCII MSH 4.1，完整三次 Lagrange 曲边三角形。
- `naca0012.svg`、`naca0012.png`：全域及 PNG 的翼型附近网格预览。
- `naca0012.json`：单元数、面积、最小采样 Jacobian 和检查记录。

弦长为 1，前缘 `(0,0)`、尾缘 `(1,0)`。厚度公式为

```text
y_t = 0.6 * (0.2969 sqrt(x) - 0.1260 x - 0.3516 x²
             + 0.2843 x³ - 0.1036 x⁴)
y_upper = y_t; y_lower = -y_t
```

采用闭合尾缘系数 `-0.1036`，上下表面各用 81 个余弦分布点拟合样条。几何和高阶网格都是公式翼型的近似。默认远场为 `[-15,21] × [-15,15]` 的矩形（前方/上下 15c、尾缘后 20c）；翼面目标尺寸 `.025`，远场尺寸 `2`，通过距离尺寸场向外变粗。这是非结构三角网格，尚未提供 C 型四边形分块或专用尾迹加密带。

Physical 标签：二维 `fluid=1`；一维 `farfield=1`、`airfoil=2`。翼型内部为空洞。生成脚本读回网格，并通过 `MeshFR2D` 检查曲边 Jacobian、面连接及法向匹配；Jacobian 检查发生在离散采样点，不能替代整个连续单元内的严格正性证明。

```bash
# 加密翼面并调整远场；输出文件名可自行指定
python case/naca0012-euler/generate_mesh.py \
  --wall-size .0125 --far-size 1 --radius 20 \
  --output case/naca0012-euler/results/mesh/fine.msh
```

几何阶数 `--order` 和求解多项式次数 `--p` 独立。

### 下游延长至 40c

保持上游和上下边界距离约 20c，仅将出口放在尾缘后 40c，即 `x=41`：

```bash
python case/naca0012-euler/generate_mesh.py \
  --radius 20 --downstream 40 --wall-size .02 --far-size 2.5 \
  --output case/naca0012-euler/results/mesh/downstream40.msh
```

新外域为 `[-20,41] × [-20,20]`；旧 `moderate.msh` 保留。翼面和远场尺寸参数不变，但 Gmsh 重新剖分后，近场单元连接并不保证与旧网格相同。`--downstream` 当前默认固定为 20c，不随 `--radius` 改变；它从尾缘而非前缘测量。上面的 40c 网格为历史测试配置。

## 计算

采用二维 Euler、Rusanov 通量，默认时间推进为显式 SSPRK(4,3)（四阶段三阶）；`--rk rk4` 可选择经典四阶 RK4。默认空间 `p=3`，积分点数参数为 6。远场使用法向特征条件，壁面采用反射状态对应的 Rusanov 壁面通量。未使用隐式推进。

以下命令是手动运行示例；当前攻角扫描已暂停，尚未得到收敛升阻力。

```bash
python case/naca0012-euler/solve.py \
  --mesh case/naca0012-euler/results/mesh/downstream40.msh \
  --alpha 0 --mach .3 --p 3 --cfl 3 --max-steps 10000 \
  --output case/naca0012-euler/results/downstream40_alpha0
```

默认已改为全局显式时间步。可选 `--time-mode local` 使用逐单元局部伪时间步，其迭代过程和累计最小伪时间不能当作真实物理时间。CFL 参数与本程序的面积/周长尺度定义配套。均匀初始流场不满足翼面无穿透条件，会产生启动扰动。

`history.csv` 保存迭代数、累计最小伪时间、CL、压力阻力及残差；`cp.csv` 保存翼面数值压力系数和 Euler 模型的 `Cf=0`；`solution.pt` 保存检查点；`metrics.json` 保存收敛状态与判据。压力采用反射 Rusanov 通量的数值壁压，以保证载荷和边界动量通量一致。流体域外法向指向翼型内部，机体压力力为 `integral((p_wall-p_inf) n_fluid ds)`。

当前默认停止条件已恢复为残差：`r <= --residual-tol` **且** `r/r_initial <= --residual-relative-tol`，默认阈值分别为 `1e-6` 和 `1e-5`。每个采样批次（默认 50 步）检查；升阻力变化不再控制停止。`--stop-criterion lift` 可复现之前的相邻完整 RK 步升力停止试验，公式为 `abs(CL_n-CL_previous)/max(abs(CL_previous),1e-12) < 1e-6`，仅此模式逐步检查。`metrics.json` 中 `convergence.criterion` 与 `criteria.stopping_rule` 记录实际采用的条件。历史结果的状态不追溯修改，解释结果时需看对应判据；后文升力停止试验不能视为残差收敛。

零攻角按新条件从 CFL=6 的第 12000 步续算，199 步后在第 12199 步停止，用时约 16 秒。相邻 CL 从 `-0.00039545760980349805` 变为 `-0.00039545721769919596`，相对变化 `9.9152e-7`；停止时压力 CD=`1.14361e-4`，残差=`2.43838e-3`。这满足新的停止条件，但近期 CL 从正值降至负值后刚开始回升，停止位置靠近波动极小值，尚不能说明气动力稳定。原有最近 1000 步 CL 极差仍为 `9.0273e-4`。结果、Cp 和诊断图保存在 `results/zero_cl_relative/`，未覆盖旧检查点。

`scan.py` 提供 0–15° 批量计算和曲线输出；当前保持停止，不应将已有启动瞬态视为收敛曲线。`diagnose_zero.py` 提供有限步数的零攻角诊断，已有数据位于 `results/diagnose_zero/`。

```bash
python -m unittest discover -s case/naca0012-euler -p test_case.py -v
```

## 当前验证与限制

网格通过仓库读入、采样 Jacobian 和共享面检查。零攻角保存到 2000 步时残差从 `0.1734` 降至 `0.01094`，仍未收敛。短对照中，将 CFL 从 3 降至 1，在接近的累计最小伪时间增量下，得到接近的残差和载荷；这不能排除远场反射，也不能证明最终稳态收敛。下游延长网格已完成 10000 步全局显式时间步诊断，t=6.7533；残差降至 0.003204，仍未通过稳态判据，详见 [诊断报告](diagnostic_report.md)。

Euler 不包含黏性边界层、湍流和常规黏性失速，CD 仅包含压力力。框架尚无激波限制器和正性保持器。还需独立检查边界影响、网格敏感性和零攻角对称性。

## 三角形网格质量

```bash
python case/naca0012-euler/mesh_quality.py \
  --mesh case/naca0012-euler/results/mesh/downstream40.msh \
  --output case/naca0012-euler/results/mesh_quality_downstream40
```

输出 `quality.json`、逐单元 `cells.csv`、`quality.png` 和 `report.md`。检查顶点三角形最小角、等边归一化形状质量 `4 sqrt(3) A / sum(edge²)`、归一化最长边/对应高度比，以及高阶几何的 Jacobian 最小值、单元内变化比和缩放 Jacobian。每单元采用 325 个参考三角形均匀格点加 64 个体积分点，共 389 个采样点。

下游 40c 网格：最小角 34.15°，最低形状质量 0.734，中位数 0.991，最大归一化长宽比 2.29，最低 Jacobian 最小/最大比 0.606，无非正采样 Jacobian。相邻单元等效尺寸比（面积比开平方）的最大值为 1.523，没有超过 2 的跳变。原 20c 网格最小角为 33.58°、最低形状质量 0.725，同样未发现翻转或明显扁瘦单元。质量指标不能证明流动分辨率足够，也不能代替稳定性验证。

## 全局显式时间步诊断运行

为观察压力扰动的传播时间，使用 `--time-mode global`，将每步所有单元的步长统一为 CFL 估计的最小值；所选显式 RK 方法的四个阶段使用同一步长。默认 `global` 表示全局物理时间推进；`local` 是可显式选择的局部伪时间推进。`--quadrature 6` 指每方向 6 个 Gauss 点：三角形 Duffy 体积分 36 点、每条边 6 点；`p=3` 的每变量解自由度始终为 10，不受积分规则改变影响。

```bash
python case/naca0012-euler/solve.py \
  --mesh case/naca0012-euler/results/mesh/downstream40.msh \
  --alpha 0 --p 3 --quadrature 6 --cfl 3 --time-mode global \
  --diagnostics --max-steps 10000 --snapshot-every 1000 \
  --output case/naca0012-euler/results/downstream40_global0

python case/naca0012-euler/plot_diagnostics.py \
  case/naca0012-euler/results/downstream40_global0
```

`--diagnostics` 默认每 50 步追加并刷新以下文件：

- `probes.csv`：翼型上下方、尾迹及靠近四周边界的固定探针处的 rho、u、v、p、p'、Mach 和熵代理量。
- `probes.json`：请求坐标与实际选取的固定解点坐标、距离。探针取最近解点，不进行连续插值。
- `diagnostics.csv`：近场/尾迹/远场物理积分 RMS 残差、最大残差单元坐标、最小密度/压力、最大 Mach、壁面内部迹的法向速度误差、数值壁压与内部压力之差、质量/能量及瞬时全局通量收支、实际时间步。
- `boundary.csv`：分别记录左/右/上/下边界的压力扰动、法向线性声学出入射指标、质量及能量通量。声学指标用于找传播线索，不能直接视为多维反射系数。
- `snapshots/`：每 1000 步保存解状态 PT，以及单元平均压力、速度、Mach、熵代理量和单元残差 CSV。

`plot_diagnostics.py` 生成诊断总图、边界声学指标图、最近流场图和 `diagnostic_summary.json`。瞬时守恒误差定义为 `integral(rhs)+integral(boundary_numerical_flux)`，应接近舍入误差；它与累计质量变化、时间积分误差是不同指标。使用 Ctrl-C 会在完成当前完整 RK 步后保存状态并记为 `stopped_by_user`。跨网格检查点会拒绝直接重启；切换时间模式时保留状态但重新累计时间和历史。


## SSPRK(4,3) 与 CFL 试验

按用户要求，后续默认时间推进改为 `--rk ssprk43`；之前 10000 步诊断及 `diagnostic_report.md` 中的历史结果仍属于经典 RK4，不能重新标记为 SSPRK(4,3)。

本算例的 CFL 定义为 `dt = CFL * min_K[A_K / ((2*p+1)*P_K*max_K(|u|+c))]`，`p=3` 的阶数因子为 7。A、P 分别来自曲边面积和周长积分；每步重新估计声速和速度。这个 CFL 数与按边长定义的 Courant 数不同。空间/时间阶数不能单独决定允许的最大步长。

```bash
python case/naca0012-euler/test_cfl.py
```

试验从 `downstream40_global0/solution.pt` 的同一个状态出发，加入相同随机种子、幅度 `1e-8` 的扰动，分别用 SSPRK(4,3) 和 CFL=3/4/6/8 推进相同的物理时间增量 0.25。报告位于 `results/ssprk43_cfl_test/`。短时通过不是长时间稳定性证明，还需较长续算验证。改变时间推进方法重启时会重新记录该段历史，保留原始残差判据基准和累计物理时间；命令行 CFL 会覆盖检查点中旧 CFL。

四组短测均通过，CFL=8 相对 CFL=3 的最终状态分量 RMS 差最大为 `2.36e-11`。随后选择 CFL=6，从未加扰动的第 10000 步检查点续算 2000 步：

```bash
python case/naca0012-euler/solve.py \
  --mesh case/naca0012-euler/results/mesh/downstream40.msh \
  --alpha 0 --p 3 --quadrature 6 --rk ssprk43 --time-mode global --cfl 6 \
  --restart case/naca0012-euler/results/downstream40_global0/solution.pt \
  --max-steps 12000 --diagnostics --snapshot-every 1000 \
  --output case/naca0012-euler/results/downstream40_ssprk43_cfl6
```

该段完成用时约 148 秒，实际步长约 `1.370e-3`，为之前 CFL=3 的两倍；累计物理时间从 6.75328 增至 9.49273，没有回退或降低 CFL。残差从 `3.204e-3` 降至 `2.493e-3`（下降约 22%）。末值 CL=`1.352e-4`、压力 CD=`-4.426e-5`；最近 1000 步 CL/CD 极差分别为 `9.847e-4`、`2.049e-4`，**仍未达到稳态收敛判据，不能作为最终气动力**。

近场残差从 `6.865e-4` 降至 `3.769e-4`，尾迹残差降至 `2.644e-4`；外区残差增至 `1.775e-3`，与启动扰动向外传播相符，但仅凭此不能确定传播机制。四周边界压力扰动 RMS 最大约 `9.96e-10`，本时段尚无明显边界扰动证据。瞬时质量/能量通量收支误差最大约 `2.83e-13`/`7.08e-13`。详细记录、检查点和图在 `results/downstream40_ssprk43_cfl6/`；这段验证支持当前算例使用 CFL=6，不能代表任意攻角或任意长时间的稳定性保证。

进一步加倍至 CFL=12 未通过稳定性测试。以第 12000 步状态加同种子 `1e-8` 扰动为共同初值，目标物理时间增量为 1：CFL=6 用 731 步完成，残差从 `2.493e-3` 降至 `2.310e-3`；CFL=12 的初始时间步为 `2.739e-3`，仅推进 7 步（时间增量约 `0.0191`）便未通过密度/压力正性或有限性检查，提前终止。因此当前保留 CFL=6；失败测试没有覆盖原有检查点。报告在 `results/ssprk43_cfl12_test/report.json`，复现命令为：

```bash
python case/naca0012-euler/test_cfl.py \
  --checkpoint case/naca0012-euler/results/downstream40_ssprk43_cfl6/solution.pt \
  --cfls 6 12 --duration 1 \
  --output case/naca0012-euler/results/ssprk43_cfl12_test
```

## 减少网格的同残差对照

默认停止条件恢复为残差后，两套网格均从均匀来流开始，使用 p=3、每方向 6 点积分、显式 SSPRK(4,3)、全局 CFL=6 及相同启动 CFL 渐增规则，依次运行以避免相互争抢 CPU。初步门槛设为绝对残差 `1e-2`，相对残差上限为 1（只要求不超过初值）；每 50 步检查一次。这是启动阶段的性能比较，不代表严格稳态或相同精度。

| 网格 | 三角形数 | 到达门槛步数 | 实际耗时 | 无量纲物理时间 t a_inf/c | 最终残差 |
|---|---:|---:|---:|---:|---:|
| 原网格 | 5574 | 1200 | 89.9 秒 | 1.07060 | 9.919e-3 |
| 粗网格 | 2810 | 800 | 31.7 秒 | 0.85224 | 9.815e-3 |

粗网格单元数减少约 50%，本次到达同一绝对残差门槛的实际运行速度提高约 2.83 倍。末步时间步由 `1.365e-3` 增至 `1.767e-3`，两组均无拒绝批次。计时来自求解器的 elapsed_seconds，含编译缓存加载与诊断，不含网格生成和进程导入。粗网格初始残差为 0.11795，原网格为 0.14667；不同离散网格的同残差不保证相同 Cp/气动力误差，也不能据此外推严格收敛耗时。

粗网格保留上游/上下 20c、下游 40c 和三阶曲边几何，壁面尺寸 `.03`、远场尺寸 `3.5`。最小角 31.4 度、最小形状质量 0.6768、最小 Jacobian 比 0.4965，所有 Jacobian 采样为正。另生成过壁面 `.04` 的 2290 单元候选网格，但曲边 Jacobian 比降至 0.165，未选作对照。

```bash
python case/naca0012-euler/generate_mesh.py \
  --radius 20 --downstream 40 --wall-size .03 --far-size 3.5 --order 3 \
  --output case/naca0012-euler/results/mesh/coarse30_40.msh

python case/naca0012-euler/solve.py \
  --mesh case/naca0012-euler/results/mesh/coarse30_40.msh \
  --alpha 0 --p 3 --quadrature 6 --rk ssprk43 --time-mode global --cfl 6 \
  --stop-criterion residual --residual-tol .01 --residual-relative-tol 1 \
  --max-steps 12000 --diagnostics \
  --output case/naca0012-euler/results/compare_residual_coarse
```

原网格对照将上述 mesh 改为 `downstream40.msh`，output 改为 `compare_residual_fine`，不使用重启。汇总和残差曲线位于 `results/mesh_residual_comparison/`，每套计算各自保存 Cp、检查点和诊断图；旧结果未覆盖。

粗网格严格残差试算从上述 800 步检查点继续，保留最初均匀来流的残差基准 `0.11794878`，不以续算起点重置相对残差。绝对阈值 `1e-6`、相对阈值 `1e-5`，两者同时满足才停止；此次相对条件对应绝对残差约 `1.179e-6`，因此绝对 `1e-6` 是更严格的约束。最多累计 100000 步，每 500 步保存检查点，每 5000 步保存流场快照。实时状态查看 `results/coarse40_strict0/metrics.json` 与 `results/coarse40_strict0.log`，不能以启动记录代替最终结果。

```bash
python case/naca0012-euler/solve.py \
  --mesh case/naca0012-euler/results/mesh/coarse30_40.msh \
  --alpha 0 --p 3 --quadrature 6 --rk ssprk43 --time-mode global --cfl 6 \
  --stop-criterion residual --residual-tol 1e-6 --residual-relative-tol 1e-5 \
  --restart case/naca0012-euler/results/compare_residual_coarse/solution.pt \
  --max-steps 100000 --diagnostics --snapshot-every 5000 \
  --output case/naca0012-euler/results/coarse40_strict0
```

### 粗网格第 100000 步的 CFL=10 稳定性试验

CFL=6 严格残差试算累计到 100000 步后达到步数上限，续算段用时 3293 秒（约 55 分钟），残差 `2.31885e-6`，相对初始残差 `1.96598e-5`，未满足严格阈值；CL=`1.82035e-4`、压力 CD=`1.91936e-4`。最近 1000 步 CL/CD 极差约 `4.23e-7`/`1.16e-7`，近场残差 `2.178e-7`，外区残差 `2.483e-6`。

在该检查点上加入相同种子、幅度 `1e-8` 的扰动，用 CFL=6 和 10 分别测试目标物理时间增量 1。CFL=6 用 470 步稳定完成，残差约 `2.307e-6`；CFL=10 初始步长约 `3.550e-3`，第 14 步（物理时间增量约 0.04954）未通过流场正性/有限性检查，提前终止，未启动 CFL=10 的长期续算。原检查点未被覆盖。报告位于 `results/coarse_cfl10_screen/report.json`。

```bash
python case/naca0012-euler/test_cfl.py \
  --checkpoint case/naca0012-euler/results/coarse40_strict0/solution.pt \
  --cfls 6 10 --duration 1 \
  --output case/naca0012-euler/results/coarse_cfl10_screen
```

### 粗网格 CFL=8 验证

同一第 100000 步检查点加相同 `1e-8` 扰动后，CFL=6/8 均稳定推进物理时间增量 1，分别用 470/353 步、18.26/13.65 秒；最终状态分量 RMS 差最大约 `1.10e-12`。短测报告为 `results/coarse_cfl8_screen/report.json`。

随后从未加扰动的第 100000 步原检查点用 CFL=8 续算 2000 步（其余设置与严格残差试算相同），完成耗时 70.24 秒，没有回退或降低 CFL。累计物理时间增至 217.78662，残差从 `2.31885e-6` 降至 `2.25449e-6`，CL=`1.81895e-4`、压力 CD=`1.91680e-4`，仍未满足严格残差条件。CFL=8 比 6 的实际步长大约 33%，同物理时间短测耗时减少约 25%；该结果只验证当前状态附近的稳定性。检查点和诊断保存在 `results/coarse40_cfl8/`，原 CFL=6 检查点保留。

## 单个 case 内的 CPU 并行与当前攻角扫描

计算热路径进一步优化：默认将 CFL 步长计算与完整 RK 步一起通过 `torch.compile(fullgraph=True)` 编译，CFL 使用可更新的标量张量，避免启动 ramp 或回退改变 CFL 时按数值重复编译。`--eager` 同时关闭 RHS 和完整时间步编译。RK 内部保留 `[单元,节点,变量]` 布局，逐单元步长直接广播，不再展开为完整状态大小。首次运行包含编译开销，短算例应单独评估总耗时。

面通量直接计算法向 Euler 通量，并复用速度、压力和声速；壁压仅在翼面求值。通用 `MeshFR2D` 和翼型专用 RHS 都使用初始化时生成的邻居面/积分点索引，通过一次 gather 完成面方向对齐。`benchmark_parallel.py` 已改为预热并测量完整编译步，输出中的 `compiled_full_step` 标明新口径；下文旧基准保留为历史记录。

本次修改前后在同一 2810 单元检查点、8 线程下顺序运行，预热后各重复 3 次、每次 40 步（含每步载荷计算），完整步中位数由 23.72 ms 降至 13.19 ms，约加速 1.80 倍；独立 RHS 中位数分别为 4.88/4.99 ms，没有显示出加速，主要收益来自完整步编译。40 步后的最大状态差为 `6.0e-15`，两组状态均有效，改变 CFL 未产生额外编译图。报告保存在 `results/compute_optimization/{before,after}.json`。这是本次独立对照，不能与下文历史测试的绝对耗时直接比较。

单个 case 的批量单元/面运算由 PyTorch CPU 内核并行执行。`solve.py --threads 8` 设置算子内部线程数，默认 8；`scan.py --workers 1 --threads 8` 表示一次运行一个攻角，每个攻角内部 8 线程。两层并行可独立设置，线程数不会改变离散公式或 CFL。

`benchmark_parallel.py` 在独立子进程中从同一检查点开始，对每种线程数先预热，排除编译时间，重复 3 次，每次推进 40 步，比较相同的 dt/RK/载荷计算。i7-14700、2810 单元的单步中位数为：1/2/4/8/16 线程分别约 33.0/22.0/15.8/13.6/15.3 ms。8 线程相对原始单线程的 33.8 ms 约快 2.48 倍。线程间最终状态相对原始单线程的最大绝对差不超过 `1.86e-14`。完整 8 线程求解器另完成 500 步续算验证，耗时 7.69 秒，无回退。7 项算例测试通过，包含与仓库残差实现、原始翼面迹求值的一致性检查。

实现中预先将面提升矩阵整理成连续批量矩阵，避免每次 RHS 的重复排列；载荷仅计算翼面迹。残差停止模式不再每步计算无关的升力停止条件。基准原始数据在 `results/parallel_baseline/`、`results/parallel_optimized/`、`results/parallel_16/`，汇总为 `results/parallel_comparison.json`。这些性能数字只代表该网格及本机本次测试。

当前授权扫描采用绝对残差 `<=1e-5`，相对残差上限设为 1，保留粗网格、p=3、6 点积分、显式 SSPRK(4,3)、全局 CFL=8。按 0–15 度、每隔 1 度逐个推进，可用前一已收敛攻角热启动。每个攻角最多 200000 步，若未通过残差条件，则在汇总中标记未收敛且不输出为最终值。`metrics.json` 会记录真实 CFL 及回退次数；若发生不稳定，求解器仍按既有机制减小 CFL，不隐瞒实际值。

来流方向为 `(cos(alpha),sin(alpha))`，升力方向为 `(-sin(alpha),cos(alpha))`。翼面压力积分得到 `(Cx,Cy)` 后，`Cd_pressure=Cx*cos(alpha)+Cy*sin(alpha)`，`Cl=-Cx*sin(alpha)+Cy*cos(alpha)`；升力的负 x 分量不单独计为阻力。Cf 是 Euler 模型的零摩擦系数，不代表压力 Cd 必须为零。

```bash
python case/naca0012-euler/scan.py \
  --mesh case/naca0012-euler/results/mesh/coarse30_40.msh \
  --output case/naca0012-euler/results/polar_coarse8_threads8_r1e5 \
  --start 0 --end 15 --workers 1 --threads 8 --max-steps 200000 --cfl 8 \
  --residual-tol 1e-5 --residual-relative-tol 1 \
  --initial-restart case/naca0012-euler/results/polar_coarse8_threads8_r1e5/alpha_00/solution.pt
```

0 度直接复用已经低于该门槛的检查点；后续攻角运行中。每完成一个攻角会更新输出目录内 `polar.csv`、`report.md`、`polar.png` 以及 Cp/收敛曲线，最终状态以各攻角的 `metrics.json` 为准。运行日志为 `results/polar_coarse8_threads8_r1e5.log` 和该输出目录内各 `alpha_XX.log`。

扫描后续按用户要求切换为 `--workers 2 --threads 8`：两个独立攻角同时运行，每个 case 内部 8 线程。调度采用共享待算队列，一个完成后即领取下一个攻角，避免固定分组导致某个工作槽提前空闲；待算数量不足两个时自然减少并发。每个工作槽保留自己最近的已收敛解供后续攻角热启动，已有同攻角检查点优先续算。输出目录和残差门槛不变，已完成的 0 度不会重算。

用户随后将扫描范围缩减为 **0–5 度**：6 度在第 15500 步检查点停止并标记 `stopped_by_user`，取消 7–15 度；仅保留 5 度从第 86500 步检查点继续。当前调度为 `--start 0 --end 5 --workers 1 --threads 8`，其余参数与上文相同。5 度完成后自动退出并更新 0–5 度的曲线和汇总；6 度的未收敛检查点只作存档，不计入最终曲线。

## 最新网格范围：前/上/下 15c、后 20c

生成器的默认外域已改为 `x/c ∈ [-15,21]`、`y/c ∈ [-15,15]`：上游从前缘量 15c，下游从后缘量 20c。沿用当前粗网格的壁面尺寸 `.03`、远场尺寸 `3.5` 和三阶曲边几何，另存为 `results/mesh/coarse15_down20.msh`，共 2020 个三角形（p=3 时每变量 20200 个自由度），相比旧 2810 单元减少约 28%。新网格重新剖分，近场连接不保证与旧网格一致。

```bash
python case/naca0012-euler/generate_mesh.py \
  --radius 15 --downstream 20 --wall-size .03 --far-size 3.5 --order 3 \
  --output case/naca0012-euler/results/mesh/coarse15_down20.msh
python case/naca0012-euler/mesh_quality.py \
  --mesh case/naca0012-euler/results/mesh/coarse15_down20.msh \
  --output case/naca0012-euler/results/mesh_quality_coarse15_down20
```

读回、曲边面匹配检查通过；最小角 29.39 度、最小形状质量 0.6836、最小 Jacobian 比 0.5379，所有 Jacobian 采样为正，相邻单元尺寸比没有超过 2。网格 PNG/JSON 与 MSH 同目录，详细质量记录在 `results/mesh_quality_coarse15_down20/`。当前 5 度继续使用旧网格完成，既有 0–5 度结果仍属于旧外域；此操作没有启动新网格求解，也不能把旧检查点直接用于新网格重启。
