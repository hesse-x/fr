# Gmsh 网格生成

入口：[mesh_generation.py](../mesh_generation.py)。以下命令均从仓库根目录执行。

本工具生成贴体介电圆柱的曲边四边形/三角形网格，或一般不规则四边形网格。
它只生成几何和网格，不求解物理场；实际读入网格后的平流验证见[mesh_advection](../mesh_advection/README.md)。

需要PATH中存在`gmsh`命令，不要求安装Gmsh Python包。框架支持ASCII Gmsh MSH 2.2/4.1，
保留Physical区域与边界标签。几何阶数、网格类型及输出位置由命令行参数指定。

```bash
# 多块结构化三次曲边四边形，圆柱内外共形。
python -m case.mesh_generation --kind cylinder-quad --order 3 --output results/mesh/cylinder_quad.msh

# 无结构三次曲边三角形。
python -m case.mesh_generation --kind cylinder-tri --order 3 --output results/mesh/cylinder_tri.msh

# 一般不规则四边形，导出ASCII MSH 2.2。
python -m case.mesh_generation --kind irregular-quad --version 2.2 --output results/mesh/irregular.msh

python -m case.mesh_generation --help
```

读入已有网格无需调用Gmsh。网格格式、曲边几何和离散接口见[框架说明](../../doc/framework.md)。
圆柱散射应用见[mie_cylinder](../mie_cylinder/README.md)。
