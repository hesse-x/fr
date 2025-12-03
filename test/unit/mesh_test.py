import pytest
import torch
import numpy as np
import os
from typing import Dict, List
from tempfile import NamedTemporaryFile
from mesh import *
from scipy.special import roots_legendre

# ======================= 工具函数 =======================
def interp_poly(order: int):
    p = order + 1
    root_nodes, _ = roots_legendre(p)
    Ls = []
    for i in range(p):
        roots = np.hstack([root_nodes[:i], root_nodes[i+1:]])
        coeffs = np.poly(roots)
        li = np.poly1d(coeffs)
        li = li / li(root_nodes[i])
        Ls.append(li)
    return Ls
# ===================== 测试数据准备 =====================
def create_test_msh_content() -> str:
    """构造合法的1D MSH 2.2测试文件内容"""
    return """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0.0 0.0 0.0
2 1.0 0.0 0.0
$EndNodes
$Elements
3
1 1 1 0 1 2
2 15 2 101 1 1
3 15 2 102 2 2
$EndElements
$End
"""

def create_invalid_version_msh() -> str:
    """构造不支持版本的MSH文件"""
    return """$MeshFormat
1.0 0 8
$EndMeshFormat
$End
"""

def create_invalid_node_msh() -> str:
    """构造节点格式错误的MSH文件"""
    return """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
1
1 0.0 0.0
$EndNodes
$End
"""

def create_invalid_element_msh() -> str:
    """构造单元节点数不匹配的MSH文件"""
    return """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
$EndNodes
$Elements
1
1 1 1 0 1 2 3
$EndElements
$End
"""

# ===================== 核心函数测试 =====================
class TestCoreFunctions:
    """测试核心辅助函数"""

    def test_is_boundary(self):
        """测试物理ID边界判断"""
        # 正常场景
        assert is_boundary(0) is False
        assert is_boundary(99) is False
        assert is_boundary(100) is True
        assert is_boundary(199) is True
        
        # 异常场景：非法ID
        with pytest.raises(AssertionError) as excinfo:
            is_boundary(-1)
        assert "overflow" in str(excinfo.value)
        
        with pytest.raises(AssertionError) as excinfo:
            is_boundary(200)
        assert "overflow" in str(excinfo.value)

    def test_cell_class(self):
        """测试Cell类初始化和字符串表示"""
        node_ids = torch.tensor([0, 1], dtype=torch.int64)
        cell = Cell(phy_id=0, elem_type=1, node_ids=node_ids)
        
        # 属性验证
        assert cell.phy_id == 0
        assert cell.elem_type == 1
        assert torch.equal(cell.node_ids, node_ids)
        
        # 字符串表示验证（基础）
        cell_str = str(cell)
        assert "Cell(" in cell_str
        assert "phy_id: 0" in cell_str
        assert "elem_type: 1" in cell_str
        assert "[0, 1]" in cell_str

    def test_mesh_class(self):
        """测试Mesh类初始化和字符串表示"""
        mesh = Mesh()
        # 初始状态
        assert mesh.dim == 0
        assert mesh.nodes.shape == torch.Size([0])
        assert len(mesh.inner) == 0
        assert len(mesh.boundary) == 0
        
        # 添加测试数据
        node_ids = torch.tensor([0, 1], dtype=torch.int64)
        inner_cell = Cell(phy_id=0, elem_type=1, node_ids=node_ids)
        boundary_cell = Cell(phy_id=101, elem_type=1, node_ids=torch.tensor([0], dtype=torch.int64))
        mesh.nodes = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        mesh.inner.append(inner_cell)
        mesh.boundary.append(boundary_cell)
        mesh.dim = 1
        
        # 属性验证
        assert mesh.dim == 1
        assert mesh.nodes.shape == (2, 3)
        assert len(mesh.inner) == 1
        assert len(mesh.boundary) == 1
        
        # 字符串表示验证
        mesh_str = str(mesh)
        assert "Mesh:" in mesh_str
        assert "dim: 1" in mesh_str
        assert "Nodes number: 2" in mesh_str
        assert "Inner cells number: 1" in mesh_str
        assert "Boundary cells number: 1" in mesh_str

# ===================== MSH解析测试 =====================
class TestMSHParser:
    """测试MSH文件解析功能"""

    def test_parse_valid_msh(self, tmp_path):
        """测试解析合法的MSH文件"""
        # 创建临时MSH文件
        msh_file = tmp_path / "test.msh"
        msh_file.write_text(create_test_msh_content())
        
        # 解析文件
        mesh = parse_msh(str(msh_file))
        
        # 基础属性验证
        assert mesh.dim == 1  # 单元类型1对应1D
        assert mesh.nodes.shape == (2, 3)  # 2个节点，3维坐标
        assert torch.equal(mesh.nodes, torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64))
        
        # 内场单元验证
        assert len(mesh.inner) == 1
        inner_cell = mesh.inner[0]
        assert inner_cell.phy_id == 0
        assert inner_cell.elem_type == 1
        assert torch.equal(inner_cell.node_ids, torch.tensor([0, 1], dtype=torch.int64))
        
        # 边界单元验证
        assert len(mesh.boundary) == 2
        boundary_cell1 = mesh.boundary[0]
        assert boundary_cell1.phy_id == 101
        assert torch.equal(boundary_cell1.node_ids, torch.tensor([0], dtype=torch.int64))
        
        boundary_cell2 = mesh.boundary[1]
        assert boundary_cell2.phy_id == 102
        assert torch.equal(boundary_cell2.node_ids, torch.tensor([1], dtype=torch.int64))

    def test_parse_invalid_version(self, tmp_path):
        """测试解析不支持版本的MSH文件"""
        msh_file = tmp_path / "invalid_version.msh"
        msh_file.write_text(create_invalid_version_msh())
        
        with pytest.raises(ValueError) as excinfo:
            parse_msh(str(msh_file))
        assert "不支持的MSH版本" in str(excinfo.value)
        assert "1.0" in str(excinfo.value)

    def test_parse_invalid_node(self, tmp_path):
        """测试解析节点格式错误的MSH文件"""
        msh_file = tmp_path / "invalid_node.msh"
        msh_file.write_text(create_invalid_node_msh())
        
        with pytest.raises(ValueError) as excinfo:
            parse_msh(str(msh_file))
        assert "节点格式错误" in str(excinfo.value)
        assert "应为ID x y z" in str(excinfo.value)

    def test_parse_invalid_element(self, tmp_path):
        """测试解析单元节点数不匹配的MSH文件"""
        msh_file = tmp_path / "invalid_element.msh"
        msh_file.write_text(create_invalid_element_msh())
        
        with pytest.raises(ValueError) as excinfo:
            parse_msh(str(msh_file))
        assert "单元类型 1 节点数不匹配" in str(excinfo.value)
        assert "实际3个，预期2个" in str(excinfo.value)

# ===================== 矩阵计算测试 =====================
class TestMatrixComputation:
    """测试插值矩阵和导数矩阵计算"""

    def test_compute_interp_interface_matrix(self):
        """测试界面插值矩阵计算"""
        # 构造1D线段单元
        cell = Cell(phy_id=0, elem_type=1, node_ids=torch.tensor([0, 1], dtype=torch.int64))
        
        # 1阶（order=1，2个节点）
        interp_mat = compute_interp_interface_matrix(cell, order=1)
        assert interp_mat.shape == (2, 2)  # [2个界面点, 2个单元节点]
        # 验证插值矩阵值（Legendre根在order=1时是[-√3/3, √3/3]，插值到-1和1）
        Ls = interp_poly(order=1)
        target = torch.tensor([[Ls[0](-1), Ls[1](-1)], [Ls[0](1), Ls[1](1)]])
        assert torch.allclose(interp_mat, target)
        
        # 2阶（order=2，3个节点）
        interp_mat_2 = compute_interp_interface_matrix(cell, order=2)
        assert interp_mat_2.shape == (2, 3)
        Ls = interp_poly(order=2)
        target_2 = torch.tensor([[Ls[0](-1), Ls[1](-1), Ls[2](-1)], [Ls[0](1), Ls[1](1), Ls[2](1)]])
        assert torch.allclose(interp_mat_2, target_2)

    def test_compute_deri_matrix(self):
        """测试导数矩阵计算"""
        # 构造1D线段单元
        cell = Cell(phy_id=0, elem_type=1, node_ids=torch.tensor([0, 1], dtype=torch.int64))
        
        # 1阶（order=1，2个节点）
        deri_mat = compute_deri_matrix(cell, order=1)
        assert deri_mat.shape == (2, 2)  # [节点数, 节点数]
        
        # 验证导数矩阵（Legendre根[-√3/3, √3/3]的导数矩阵）
        Ls = interp_poly(order=1)
        root_nodes, _ = roots_legendre(2)
        Ls = [li.deriv() for li in Ls]
        expected_deri = torch.tensor([[Ls[0](root_nodes[0]), Ls[1](root_nodes[0])],
                                      [Ls[0](root_nodes[1]), Ls[1](root_nodes[1])]], dtype=torch.float64)
        assert torch.allclose(deri_mat, expected_deri, atol=1e-6)
        
        # 2阶（order=2，3个节点）
        deri_mat_2 = compute_deri_matrix(cell, order=2)
        assert deri_mat_2.shape == (3, 3)

        Ls = interp_poly(order=2)
        Ls = [li.deriv() for li in Ls]
        root_nodes, _ = roots_legendre(3)
        expected_deri_2 = torch.tensor([[Ls[0](x) for x in root_nodes],
                                        [Ls[1](x) for x in root_nodes],
                                        [Ls[2](x) for x in root_nodes]]).transpose(1, 0)
        assert torch.allclose(deri_mat_2, expected_deri_2, atol=1e-6)

# ===================== 网格分组测试 =====================
class TestMeshGroups:
    """测试网格分组和界面映射"""

    @pytest.fixture
    def test_mesh(self, tmp_path):
        """构造测试用Mesh对象"""
        msh_file = tmp_path / "test.msh"
        msh_file.write_text(create_test_msh_content())
        return parse_msh(str(msh_file))

    def test_get_interface_map(self, test_mesh):
        """测试界面映射计算"""
        elem_to_interface, boundary_to_interface = get_interface_map(test_mesh)
        
        # 内场单元到界面映射（1个内场单元，2个界面）
        assert elem_to_interface.shape == (1, 2, 2)
        # 单元0的左界面是节点0（类型1），右界面是节点1（类型0）
        assert torch.equal(elem_to_interface[0, 0], torch.tensor([0, 1]))
        assert torch.equal(elem_to_interface[0, 1], torch.tensor([1, 0]))
        
        # 边界单元到界面映射（2个边界单元）
        assert boundary_to_interface.shape == (2, 2)
        assert torch.equal(boundary_to_interface[0], torch.tensor([0, 0]))  # 边界101对应节点0，类型0
        assert torch.equal(boundary_to_interface[1], torch.tensor([1, 1]))  # 边界102对应节点1，类型1

    def test_get_mesh_groups(self, test_mesh):
        """测试网格分组属性计算"""
        order = 1
        elem_props, boundary_props = get_mesh_groups(test_mesh, order)
        
        # 内场属性验证
        assert len(elem_props) == 1
        elem_prop = elem_props[0]
        assert elem_prop.elem_type == 1
        assert elem_prop.elem_ids.shape == (1,)
        assert elem_prop.interp_inteface_matrix.shape == (2, 2)  # 转置后
        assert elem_prop.deri_matrix.shape == (2, 2)  # 转置后
        assert elem_prop.elem_sizes.shape == (1,)
        assert torch.equal(elem_prop.elem_sizes, torch.tensor([1.0], dtype=torch.float64))
        assert elem_prop.elem_to_interface.shape == (1, 2, 2)
        
        # 边界属性验证
        assert len(boundary_props) == 1
        boundary_prop = boundary_props[0]
        assert boundary_prop.elem_type == 15
        assert boundary_prop.phy_id == 101
        assert boundary_prop.elem_ids.shape == (2,)
        assert boundary_prop.boundary_to_interface.shape == (2, 2)

    def test_get_interface_property(self, test_mesh):
        """测试界面属性计算"""
        interface_props = get_interface_property(test_mesh)
        
        assert len(interface_props) == 1
        interface_prop = interface_props[0]
        assert interface_prop.elem_type == 15
        assert interface_prop.elem_ids.shape == (2,)  # 2个节点对应2个界面
        assert torch.equal(interface_prop.elem_ids, torch.tensor([0, 1], dtype=torch.int64))

# ===================== 组类测试 =====================
class TestGroupClasses:
    """测试ElementGroup和InterfaceGroup类"""

    @pytest.fixture
    def test_elem_prop(self, tmp_path):
        """构造测试用ElementGroupProperty"""
        msh_file = tmp_path / "test.msh"
        msh_file.write_text(create_test_msh_content())
        mesh = parse_msh(msh_file)
        elem_props, _ = get_mesh_groups(mesh, order=1)
        return elem_props[0]

    @pytest.fixture
    def test_interface_prop(self, tmp_path):
        """构造测试用InterfaceGroupProperty"""
        msh_file = tmp_path / "test.msh"
        msh_file.write_text(create_test_msh_content())
        mesh = parse_msh(msh_file)
        interface_props = get_interface_property(mesh)
        return interface_props[0]

    def test_element_group(self, test_elem_prop):
        """测试ElementGroup初始化和字符串表示"""
        dim_u = 2  # 2个变量
        elem_group = ElementGroup(test_elem_prop, order=1, dim_u=dim_u)
        
        # 张量形状验证
        assert elem_group.u.shape == (1, 2, 2)  # [单元数, 变量数, 节点数]
        
        # 字符串表示验证
        group_str = str(elem_group)
        assert "u:" in group_str

    def test_element_group_save_load(self, test_elem_prop, tmp_path):
        """测试ElementGroup的保存和加载"""
        dim_u = 2
        elem_group = ElementGroup(test_elem_prop, order=1, dim_u=dim_u)
        
        # 初始化测试数据
        elem_group.u = torch.ones((1, 2, 2), dtype=torch.float64)
        
        # 保存到临时文件
        save_path = tmp_path / "elem_group.pt"
        elem_group.save(str(save_path))
        assert os.path.exists(save_path)
        
        # 创建新实例并加载
        new_elem_group = ElementGroup(test_elem_prop, order=1, dim_u=dim_u)
        new_elem_group.load(str(save_path))
        
        # 验证数据一致
        assert torch.equal(new_elem_group.u, elem_group.u)
        
        # 测试加载异常
        # 维度不匹配
        wrong_dim_group = ElementGroup(test_elem_prop, order=2, dim_u=dim_u)  # order=2 → 3个节点
        with pytest.raises(ValueError) as excinfo:
            wrong_dim_group.load(str(save_path))
        assert "维度不匹配" in str(excinfo.value)
        
        # 文件不存在
        with pytest.raises(FileNotFoundError):
            new_elem_group.load(str(tmp_path / "nonexist.pt"))

    def test_interface_group(self, test_interface_prop):
        """测试InterfaceGroup初始化和字符串表示"""
        dim_u = 2
        interface_group = InterfaceGroup(test_interface_prop, order=1, dim_u=dim_u)
        
        # 张量形状验证
        assert interface_group.u.shape == (2, 2, 2)  # [界面数, 变量数, 左右]
        assert interface_group.flux.shape == (2, 2, 2)
        
        # 字符串表示验证
        group_str = str(interface_group)
        assert "InterfaceGroup:" in group_str
        assert "u:" in group_str
        assert "flux:" in group_str

# ===================== 主函数 =====================
if __name__ == "__main__":
    pytest.main(["-v", __file__])
