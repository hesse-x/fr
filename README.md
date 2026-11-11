# PyTorch Flux Reconstruction

基于PyTorch的高阶FR-DG数值求解框架，支持一至三维Cartesian网格及二维曲边三角形、四边形和混合网格。
包含Euler、线化Euler/声学、Maxwell、薛定谔和Navier–Stokes算例，默认使用CPU与FP64。

## 安装与文档

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 统一包含数值计算、绘图和声学后处理依赖；自动生成网格另需安装 `gmsh` 命令行程序。NACA0012 与圆柱算例的共享原生算子还需要支持 C++20/OpenMP 的 C++ 编译器。

- [框架设计与API](doc/framework.md)
- [方程离散与边界条件算法](doc/algorithm.md)
- [算例通用运行与可视化约定](case/README.md)
- 各算例的参数、网格、运行命令、结果和性能记录见下表中的独立README。

## 算例索引

| 路径 | 计算内容 | 说明 |
|---|---|---|
| [case/advection/](case/advection/) | 一至三维周期标量平流，验证通量、时间推进和质量守恒 | [README](case/advection/README.md) |
| [case/ringleb/](case/ringleb/) | Ringleb光滑亚声速Euler流的解析解与离散精度验证 | [README](case/ringleb/README.md) |
| [case/prandtl_meyer/](case/prandtl_meyer/) | 普朗特–迈耶膨胀扇的解析场与高阶精度验证 | [README](case/prandtl_meyer/README.md) |
| [case/isentropic_vortex/](case/isentropic_vortex/) | 二维周期等熵涡平移及误差、守恒验证 | [README](case/isentropic_vortex/README.md) |
| [case/maxwell_cavity/](case/maxwell_cavity/) | 三维PEC腔体电磁驻波与解析模态对照 | [README](case/maxwell_cavity/README.md) |
| [case/uniform_array/](case/uniform_array/) | 三维均匀电流源线阵、相位扫描/端射/差波束，PML和近远场变换，与解析阵列因子对照 | [README](case/uniform_array/README.md) |
| [case/parabolic_antenna/](case/parabolic_antenna/) | 三维焦点馈源抛物面 PEC 反射天线，native FR Maxwell、PML、近远场变换及论文方向图对比 | [README](case/parabolic_antenna/README.md) |
| [case/acoustic_pml/](case/acoustic_pml/) | 二维柱面声脉冲及随流移动PML吸收 | [README](case/acoustic_pml/README.md) |
| [case/acoustic_mean_flow_pml/](case/acoustic_mean_flow_pml/) | 固定网格背景流声学/线化Euler脉冲与现有PML的数值实验 | [README](case/acoustic_mean_flow_pml/README.md) |
| [case/mesh_generation/](case/mesh_generation/) | Gmsh贴体、曲边和不规则网格生成 | [README](case/mesh_generation/README.md) |
| [case/mesh_advection/](case/mesh_advection/) | 二维曲边三角形、四边形及混合网格上的平流 | [README](case/mesh_advection/README.md) |
| [case/mie_cylinder/](case/mie_cylinder/) | 贴体介电圆柱TMz电磁散射与圆柱级数参考对照 | [README](case/mie_cylinder/README.md) |
| [case/schrodinger/](case/schrodinger/) | 自由波包、谐振子、无限深势阱和有限势垒 | [README](case/schrodinger/README.md) |
| [case/quantum_double_slit/](case/quantum_double_slit/) | 量子双缝波包散射：远场角分布与论文曲线数字化对照 | [README](case/quantum_double_slit/README.md) |
| [case/naca0012-euler/](case/naca0012-euler/) | NACA0012无黏Euler绕流，结构四边形、native、体/面过积分及1°载荷结果 | [README](case/naca0012-euler/README.md) |
| [case/naca0012-ns/](case/naca0012-ns/) | NACA0012层流NS绕流，结构四边形、无滑移绝热壁及1°的CL/CD、Cp/Cf分布 | [README](case/naca0012-ns/README.md) |
| [case/cylinder_flow/](case/cylinder_flow/) | 二维圆柱层流绕流，默认 Re=100 非定常计算 | [README](case/cylinder_flow/README.md) |
| [case/channel-dns/](case/channel-dns/) | 恒质量流量三维槽道湍流、扰动发展及平均场统计 | [README](case/channel-dns/README.md) |
| [case/taylor_green_vortex/](case/taylor_green_vortex/) | 三向周期 NS 涡衰减，公共 native-p3、Ma=0.3及公开高分辨率参考对照；Ma=0.1/0.3均完成，含完整统计曲线及Mach数敏感性对照 | [README](case/taylor_green_vortex/README.md) |
| [case/taylor_green_vertox_les/](case/taylor_green_vertox_les/) | Taylor–Green 涡显式 Vreman LES，默认 32³ 单元，native-p3/Torch、LDG，含亚格子耗散统计及网格验证说明 | [README](case/taylor_green_vertox_les/README.md) |

## 待实现算例

以下目录仅保留目标、可复用模块和验证思路，尚无求解入口或计算结果。

| 方向 | 计划内容 | 说明 |
|---|---|---|
| 波动共振 | 双势垒隧穿或光子晶体缺陷波导/谐振腔，待选型 | [README](case/wave_resonance/README.md) |
| 声学圆柱散射 | 刚性圆柱绕射、散射分布与声学精度验证 | [README](case/acoustic_cylinder_scattering/README.md) |

## 通用测试

```bash
python -m unittest discover -s tests -v
```

算例专用测试见各自README。性能数据、误差表和图像均在对应算例目录中说明，不在此处重复。
