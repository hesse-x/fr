# 贴体介电圆柱 TMz 散射

入口：[mie_cylinder.py](../mie_cylinder.py)。以下命令均从仓库根目录执行。

`case/mie_cylinder.py` 求解二维无限长介电圆柱的 **TMz** 散射，即未知量 `(Ez,Hx,Hy)`；三维球体 Mie 散射不在本例范围。`MaxwellTMz` 是公共 Maxwell 的投影，正常入射平面波沿 +x 传播。默认半径 `.5`、波长 `2`、柱内相对介电常数 `4`、相对磁导率 `1`，外部真空；界面连续条件通过物理单位法向的阻抗 Riemann 通量处理。

默认自动生成 176 个三次曲边四边形：中心块 + 内部环形块 + 外部环形块，材料界面贴合圆周；也可读入三角形及混合网格。几何仍是高阶多项式圆弧逼近，不是精确有理圆弧，但不再使用阶梯圆柱。自备网格需覆盖相同区域，Physical Surface 1 为外介质、2 为柱内，接口标签 2 可用于几何误差检查；外边界标签 1。

```bash
python -m case.mie_cylinder --validate
# 已有曲边三角形网格
python -m case.mie_cylinder --mesh results/mesh/cylinder_tri.msh --validate --output results/mie_cylinder_tri
# 增加网格密度；延长预热以检查稳态收敛
python -m case.mie_cylinder --angular 6 --radial 12 --warmup-periods 12 --validate --output results/mie_cylinder_refined
```

求解散射场，入射波为光滑开启的单频平面波。材料容量形式为 `diag(epsilon,mu,mu) U_t + div(F_bare)=0`，材料界面两侧使用各自阻抗；散射场体源为 `(I-C) dU_inc/dt`，求质量方程后除以材料容量。无材料反差时散射源严格为零，已加入测试。圆柱之外复用现有 Cartesian 方向分裂 `SplitPML`，通过曲边积分变换到物理坐标；无阻尼区为 `[-1.5,1.5]²`，外域 `[-2.5,2.5]²`。PML 吸收散射场，解析入射场不在层内衰减，外截断为 PEC。

参考解在 `case/cylinder_exact.py` 中通过 Fourier–Bessel / Hankel 圆柱级数独立计算，约定 `exp(-i omega t)`，出射波为 `H^(1)`；这是均匀、无损圆柱的中等尺寸参数算例，当前参考实现并非针对任意大参数的通用特殊函数库。默认前 6 个周期预热，再对后 2 个周期做复振幅积分；相邻周期差用于检查残余瞬态。数值远场用内部无阻尼环带的出射柱面波系数拟合获得（角向模态 -6..6），并与级数角分布及总散射宽度对照。

已实算的默认四边形结果：散射 Ez 相对 L2 误差 **0.9468%**，总散射宽度误差 **0.2901%**，角分布相对 L2 误差 **0.8964%**。总散射宽度解析值 `3.983545`，数值拟合值 `3.995101`。此时相邻周期场变化仍为 `1.28%`，精度受残余瞬态、网格、PML 和远场拟合共同影响，不能据此宣称稳态网格收敛。

输出包括 `mesh.svg`、`field.csv`（散射 Ez 复振幅）、`history.csv`、`scattering_width.csv`（解析/数值角分布）、`metrics.json`、`solution.pt`。默认自动生成时还保留 `.geo` 和 `.msh`，方便用 Gmsh 检查几何及区域标签。

另已用 Gmsh 生成的 **282 个无结构三次曲边三角形**（基础尺寸 `h=.5`）完成相同 8 周期计算：散射 Ez 相对 L2 误差 **0.8436%**，总散射宽度误差 **0.0980%**，角分布相对 L2 误差 **0.8307%**。输出保存在 `results/mie_cylinder_tri/`。该网格与默认四边形网格的分辨率和形状不同，这两组数据用于验证两种单元都能完成散射求解，不构成严格的网格收敛对比。
