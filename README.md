# PyTorch Flux Reconstruction

基于PyTorch的高阶FR-DG数值求解框架，支持一至三维Cartesian网格及二维曲边三角形、四边形和混合网格。
包含Euler、线化Euler/声学、Maxwell、薛定谔和Navier–Stokes算例，默认使用CPU与FP64。

## 安装与文档

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-plot.txt
```

仅数值计算可安装`requirements.txt`；需要自动生成Gmsh网格时另外安装`gmsh`命令行程序。

- [框架设计与API](doc/framework.md)
- [算例通用运行与可视化约定](case/README.md)
- 各算例的参数、网格、运行命令、结果和性能记录见下表中的独立README。

## 算例索引

| 路径 | 计算内容 | 说明 |
|---|---|---|
| [case/advection.py](case/advection.py) | 一至三维周期标量平流，验证通量、时间推进和质量守恒 | [README](case/advection/README.md) |
| [case/ringleb.py](case/ringleb.py) | Ringleb光滑亚声速Euler流的解析解与离散精度验证 | [README](case/ringleb/README.md) |
| [case/prandtl_meyer.py](case/prandtl_meyer.py) | 普朗特–迈耶膨胀扇的解析场与高阶精度验证 | [README](case/prandtl_meyer/README.md) |
| [case/isentropic_vortex.py](case/isentropic_vortex.py) | 二维周期等熵涡平移及误差、守恒验证 | [README](case/isentropic_vortex/README.md) |
| [case/maxwell_cavity.py](case/maxwell_cavity.py) | 三维PEC腔体电磁驻波与解析模态对照 | [README](case/maxwell_cavity/README.md) |
| [case/acoustic_pml.py](case/acoustic_pml.py) | 二维柱面声脉冲及随流移动PML吸收 | [README](case/acoustic_pml/README.md) |
| [case/acoustic_mean_flow_pml.py](case/acoustic_mean_flow_pml.py) | 固定网格背景流声学/线化Euler脉冲与现有PML的数值实验 | [README](case/acoustic_mean_flow_pml/README.md) |
| [case/mesh_generation.py](case/mesh_generation.py) | Gmsh贴体、曲边和不规则网格生成 | [README](case/mesh_generation/README.md) |
| [case/mesh_advection.py](case/mesh_advection.py) | 二维曲边三角形、四边形及混合网格上的平流 | [README](case/mesh_advection/README.md) |
| [case/mie_cylinder.py](case/mie_cylinder.py) | 贴体介电圆柱TMz电磁散射与圆柱级数参考对照 | [README](case/mie_cylinder/README.md) |
| [case/schrodinger.py](case/schrodinger.py) | 自由波包、谐振子、无限深势阱、有限势垒和二维双缝 | [README](case/schrodinger/README.md) |
| [case/double_slit_accuracy.py](case/double_slit_accuracy.py) | 双缝条纹、透射概率和对比度的独立精度对照 | [README](case/double_slit_accuracy/README.md) |
| [case/naca0012-euler/](case/naca0012-euler/) | NACA0012翼型无黏Euler绕流 | [README](case/naca0012-euler/README.md) |
| [case/naca0012-ns/](case/naca0012-ns/) | NACA0012翼型黏性Navier–Stokes绕流 | [README](case/naca0012-ns/README.md) |
| [case/channel-dns/](case/channel-dns/) | 恒质量流量三维槽道湍流、扰动发展及平均场统计 | [README](case/channel-dns/README.md) |

## 通用测试

```bash
python -m unittest discover -s tests -v
```

算例专用测试见各自README。性能数据、误差表和图像均在对应算例目录中说明，不在此处重复。
