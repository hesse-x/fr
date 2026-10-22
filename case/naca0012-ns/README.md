# NACA0012 二维层流 Navier–Stokes

默认 `Re_c=1000`、攻角 `0°`、`Ma=0.1`，不加 RANS/LES 模型。使用完整可压缩 NS、理想气体 `gamma=1.4`、常黏度、`Pr=0.72`，静止无滑移绝热翼面。无量纲 `rho_inf=c_inf=c=1`，所以 `u_inf=Ma`、`mu=Ma/Re`。对流时间为 `t_conv=t*Ma`。

这是实验性实现，尚未得到稳态或网格无关的升阻力。二维层流 NS 结果不应称作真实三维湍流 DNS。

## 运行

从仓库根目录执行，需要 Gmsh；Matplotlib 仅用于网格预览。

```bash
venv/bin/python case/naca0012-ns/generate_mesh.py
venv/bin/python case/naca0012-ns/solve.py --max-steps 100
venv/bin/python -m unittest discover -s case/naca0012-ns -p test_case.py -v
```

每个完整 RK 步后检查残差停止条件；默认每 100 步输出残差到控制台和 `residuals.csv`，可用 `--output-every 100` 配置，设为 1 则每步输出。初始和结束状态始终输出。CL/CD 和完整检查点固定每 1,000 步、初始和结束时保存。求解默认终止对流时间为 10，但另有 10000 步上限，先达到哪个就停止。显式黏性步长可能很小，达到步数上限不代表达到目标时间或稳态。输出 `history.csv`、`metrics.json`、`solution.pt`；记录物理时间、对流时间、CL、总 CD 和残差。支持 `--restart` 同参数检查点续算；以 `--residual-tol`（默认 1e-5）检查残差停止。均匀初始流场不满足无滑移条件，起始壁面惩罚力很大，不能将其当作物理气动力。

## 网格和离散

复用相邻 Euler 算例的闭合尾缘翼型几何和特征远场、压力壁面通量，脚本通过显式文件路径加载公共实现。默认三次曲边三角形，翼面尺寸 `0.04c`、远场尺寸 `2c`、近壁 `0.1c` 范围维持细化，`0.8<x/c<4, |y/c|<0.15` 加尾迹尺寸场。上游/上下 10c，出口位于尾缘后 20c。尚无各向异性贴壁层，这只是起始网格。

空间默认 p=2，体积和面过积分；黏性项使用弱形式 BR1 梯度、平均黏性面通量和速度/温度跳跃惩罚。梯度施加无滑移边界值；绝热壁面黏性能量通量设为零。动量应力采用三维 Stokes 本构的二维速度场，热通量与应力功均进入能量方程。载荷同时积分压力和数值黏性牵引，与边界动量通量一致。全局物理时间 SSPRK(4,3)，时间步同时受声学和显式扩散限制；当前保守步长不是稳定性证明。

后续需要至少三组网格对比近壁速度、Cp、Cf 和总阻力，再检查时间步、远场距离以及 Ma=0.05 的影响。层流网格应依据边界层分辨率与收敛性判断，不能仅用 y+ 判断。

## 对照来源

Di Ilio 等，*Fluid flow around NACA 0012 airfoil at low-Reynolds numbers with hybrid lattice Boltzmann method*, Computers & Fluids 166 (2018), 200–208：
https://arxiv.org/abs/2006.10487

该文含 Re=1000 的二维不同攻角结果，及零攻角更高 Re 的计算。原文未明确报告 Ma；这里的 Ma=0.1 是本算例选择，不是论文参数。需进一步核对几何、边界、图表数据后才能作定量验证；目前未导入论文参考值。

## 已完成的验证

三项测试覆盖解析剪切应力/热通量、曲边网格常值静止梯度，以及黏性项的全局边界通量收支与绝热壁面能量通量。650 单元粗网格上完成 20 步短测，状态保持正密度正压力；只验证启动运行，不代表气动力收敛。

此前的细网格另完成 20 步：12,658 单元，p=2，75,948 个状态点。末尾对流时间仅 `1.67162e-6`，残差由 `0.302818` 降至 `0.288581`，状态有效。输出位于 `results/alpha0_re1000/`。该时刻仍是无滑移边界的启动瞬态，壁面惩罚主导载荷，CL/CD 不作物理结果引用。按当前配置显式推进到稳态的成本很高，后续需要改进显式推进效率；本次没有启动长时间后台计算。现有方程与非结构网格回归测试 12 项通过。

## 减少网格数量

默认改为 `results/mesh/coarse.msh`，保持远场范围和三次几何不变，将翼面尺寸由 0.01c 改为 0.04c，尾迹尺寸随之由 0.02c 改为 0.08c，远场尺寸由 1.5c 改为 2c。单元数由 12,658 降至 1,930（减少约 85%），p=2 时状态点由 75,948 降至 11,580。求解默认输出到 `results/alpha0_re1000_coarse/`，保留此前的细网格与短测输出。粗网格适合调试和初步计算，尚未验证气动力精度。

复现此前细网格：

```bash
venv/bin/python case/naca0012-ns/generate_mesh.py --wall-size .01 --far-size 1.5 --output case/naca0012-ns/results/mesh/fine.msh
```

## 1° 攻角显式稳态计算

```bash
venv/bin/python case/naca0012-ns/solve.py --alpha 1 --time-mode local --cfl 3 --compile --max-steps 200000 --time 100 --output case/naca0012-ns/results/alpha1_re1000_explicit
```

使用显式 SSPRK(4,3)，未使用隐式结果初始化。`--time-mode local` 仅用于求稳态的局部伪时间推进；默认 `global` 仍为全局物理时间步。输出的局部模式累计最小伪时间不代表真实流动时间。残差定义为守恒变量 RHS 的物理体积加权 RMS，取四个分量的最大值，逐步检查严格小于 `--residual-tol` 后停止。此阈值不等于网格收敛或实验验证。

输出 `wall.csv` 包含 x、y、Cp、Cf、流体外法向和归一化黏性牵引。Cf 是数值黏性机体牵引沿表面切向的有符号投影，切向指向 x 增大方向，上下表面分别使用自己的切向。载荷采用与边界通量一致的数值牵引（含壁面惩罚），`metrics.json` 记录 CL/CD 的压力、黏性分量和总量；CD_viscous 包括黏性法向应力，因此不是简单的 Cf 积分。`solution.pt` 和载荷历史固定每 1,000 步及结束时保存，可 Ctrl-C 保存最后完整步。残差图优先读取 `residuals.csv`，历史结果没有该文件时读取 `history.csv`。

## 1° 攻角计算结果（显式）

在 1,930 个三角形、p=2、Re=1000、Ma=0.1 下，从均匀初场显式局部伪时间推进至 21,000 步，首次在 500 步采样点满足残差阈值：残差 `6.53811046e-6 < 1e-5`。CL=`0.0448658531`，总 CD=`0.1207079211`，其中压力 CD=`0.0307829905`，黏性 CD=`0.0899249306`。Cf 为翼面局部分布，见 `results/alpha1_re1000_explicit/wall.csv`、`loads.png`。独立使用未编译 RHS 读回检查点，复算残差和壁面数据一致，密度/压力检查通过。

最初 4,000 步使用黏性步长系数 1，之后使用 8（仅放宽显式步长估计，未改变残差离散）；CFL 始终为 3。4/8/16 倍黏性步长短测均保持有效状态；CFL=6/9 的另行短测失稳，未用于正式结果。短测记录位于结果目录的 `explicit_*trial.json`。复现继续计算：

```bash
venv/bin/python case/naca0012-ns/solve.py --alpha 1 --time-mode local --cfl 3 --viscous-step-factor 8 --compile --residual-tol 1e-6 --max-steps 200000 --time 100 --restart case/naca0012-ns/results/alpha1_re1000_explicit/solution.pt --output case/naca0012-ns/results/alpha1_re1000_explicit_tighter
venv/bin/python case/naca0012-ns/plot_loads.py case/naca0012-ns/results/alpha1_re1000_explicit
```

本次仅满足指定残差条件，尚未验证载荷稳定或网格无关性。最近 1,000 步采样的 CL 仍在 `0.04487–0.07394` 之间变化，CD 在 `0.120499–0.121433` 之间变化。Cf/Cp 在前缘有明显单元尺度起伏，粗网格不能作为高精度表面分布。完整复算检查见 `verification.json`。

## 每步残差复用

残差范数与停止条件仍每步计算，`--output-every` 只控制控制台和 CSV 输出间隔，默认 100 步。当前状态的 RHS 同时用于残差范数和 SSPRK(4,3) 第一阶段；推进时计算另外三阶段和新状态 RHS，新状态 RHS 缓存供下一步使用。初始化计算一次 RHS，之后每个完整步仍需四次新的 RHS 计算，无需为残差输出额外重复求值。输出的 step=n 对应已完成 n 个完整步的状态，拒绝候选步时保留当前状态及其 RHS。按输出间隔打印残差并写入、刷新 `residuals.csv`，初始和结束状态强制输出；CL/CD 不逐步重复计算，保持每 1,000 步和结束时保存。按用户要求，本次改动未运行测试、性能测量或求解。
