# Taylor–Green 参考数据

数据集：Kurz, Marius; Kempf, Daniel; Blind, Marcel P.; Kopper, Patrick; Offenhäuser, Philipp;
Schwarz, Anna; Starr, Spencer; Keim, Jens; Beck, Andrea (2024),
**GALÆXI Validation: Taylor-Green Vortex**, DaRUS，版本1.1。
[DOI:10.18419/DARUS-4139](https://doi.org/10.18419/DARUS-4139)，许可 **CC BY 4.0**。

本目录实际使用作者发布的FLEXI CPU计算结果：

- 目录：`TGV_incompressible_Ma0p1/CPU/TGV_CPU_Ma0p1_Re1600_GPU_64x64x64_N7/`。
- 原始数据ID **291392**：`TGV_Re1600_TGVAnalysis.csv`，本地保存为`flexi-ma01-n7-64.csv`，
  706个时刻，t=0到20，原始数值表，未进行图片数字化或平滑。
- 参数ID **291390**：本地`parameter_flexi.ini`。
- 64³单元、p=7，即512³解点；Re=1600、Ma=0.1、μ=0.000625、`IniConstDens=T`。
- 本仓库R=1；参考R=71.42857，两者均按相同的p₀=1/(γMa²)、ρ₀=U₀=L=1初始化。
  这会改变温度数值的单位；常物性下相同Pr和比热/导热换算保持相同的动力学及热通量。

下载地址、原始元数据、SHA256和分析源码来源记录于`provenance.json`。
比较脚本读取参考前验证CSV的SHA256。

列定义依据FLEXI的[公开分析实现](https://github.com/flexi-framework/flexi/blob/master/src/testcase/taylorgreenvortex/testcase.f90)：

| 参考列 | 定义（ρ₀=1，体积平均） |
|---|---|
| `Ekin comp` | ⟨ρ|u|²/2⟩ |
| `Ekin incomp` | ⟨|u|²/2⟩ |
| `Enstrophy comp` | ⟨ρ|ω|²/2⟩ |
| `DR_S` / `Dissipation Rate Incompressible` | 2μ⟨S:S⟩ |
| `DR_Sd` | 2μ⟨Sd:Sd⟩，Sd=S−(∇·u)I/3 |
| `DR_p` | −⟨p∇·u⟩ |
| `Dissipation Rate Compressible` | `DR_Sd+DR_p` |
| `ED_S` | μ⟨|ω|²⟩ |
| `ED_D` | (4/3)μ⟨(∇·u)²⟩ |

不能把参考`Dissipation Rate Compressible`直接解释成它的离散−dK/dt。
本算例额外输出由RHS计算的−dK/dt，在图中单独标注。

[HiOCFD标准说明](https://how4.cenaero.be/sites/how4.cenaero.be/files/2025-01/BS1_TaylorGreenVortexRe1600.pdf)
采用Ma=0.1及均温初场；本次为匹配实际下载参数使用常密度初场，并按用户要求使用Ma=0.3。
Ma差异会明确记录，不能把对照误差全部归为数值离散误差。


当前密集统计对比补充：`DR_u`、`DR_Sd`、`ED_D`、最大涡量、`uprime`均直接对齐参考列。
−dK/dt参考由完整公开`Ekin comp`表使用`np.gradient(..., edge_order=2)`估计；
它有参考表时间差分误差，图例与JSON定义单独注明。
温度归一化为各自初始平均值的比值；熵诊断比较相对各自初始值的增量。
Ma=0.3与Ma=0.1仍是不同物理条件，这些变换不消除Mach数敏感性。
