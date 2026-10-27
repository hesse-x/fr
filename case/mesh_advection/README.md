# 曲边三角形、四边形及混合网格平流

入口：[mesh_advection.py](../mesh_advection.py)。以下命令均从仓库根目录执行。

网格接口和离散方法见“框架设计”。本例在生成或读入的网格上求解开放边界平流。外部Gmsh网格的生成命令见[mesh_generation](../mesh_generation/README.md)。

```bash
source venv/bin/activate
# 实际读入 Gmsh 网格运行高阶平流
python -m case.mesh_advection --mesh results/mesh/irregular.msh
# 程序生成抖动、曲边网格，并逐级加密
python -m case.mesh_advection --kind tri --n 4 --convergence
python -m case.mesh_advection --kind quad --n 4 --convergence
python -m case.mesh_advection --kind mixed --n 4 --convergence
```

不规则混合网格 `p=3, t=.05` 的实测加密结果：

| 基础每边单元数 | 三角形/四边形合计 | 状态 L2 误差 | 观测阶 |
|---|---:|---:|---:|
| 4 | 24 | 2.844e-4 | — |
| 8 | 96 | 3.240e-5 | 3.13 |
| 16 | 384 | 2.144e-6 | 3.92 |


网格生成工具说明见[mesh_generation/README.md](../mesh_generation/README.md)。
