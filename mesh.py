from typing import List, Dict, Callable, Optional, Tuple
from scipy.special import roots_legendre
from scipy.interpolate import BarycentricInterpolator
import torch
import numpy as np
import re
"""
物理ID范围定义：
- 内场: 0-99
- 边界: 100-199
"""


def is_boundary(phy_id: int) -> bool:
    """判断物理ID是否为边界"""
    assert 0 <= phy_id < 200, f"phy_id: {phy_id} overflow（1-199）"
    return phy_id > 99


class Cell:
    """单元（Cell）数据类，存储单个单元的信息"""
    def __init__(self, phy_id: int, elem_type: int, node_ids: torch.Tensor):
        self.phy_id = phy_id  # 物理组ID
        self.elem_type = elem_type  # 单元类型（对应MSH定义）
        self.node_ids = node_ids  # 单元包含的节点索引（0-based）

    def __str__(self) -> str:
        """自定义Cell对象的打印格式"""
        node_ids_str = str(self.node_ids)
        return (f"Cell(\n"
                f"  phy_id: {self.phy_id}\n"
                f"  elem_type: {self.elem_type}\n"
                f"  node_ids: {node_ids_str}\n"
                f")")


class Mesh:
    """网格（Mesh）数据类，存储整个网格的完整信息"""
    def __init__(self):
        self.dim: int = 0  # 网格维度（1/2/3）
        self.nodes: torch.Tensor = torch.tensor([])  # 节点坐标数组 (n_nodes, 3)
        self.inner: List[Cell] = []  # 单元列表（存储Cell对象）
        self.boundary: List[Cell] = []  # 单元列表（存储Cell对象）

    def __str__(self) -> str:
        """自定义Mesh对象的打印格式"""
        basic_info = f"Mesh:\ndim: {self.dim}\n"
        basic_info += f"Nodes number: {self.nodes.shape[0]}\n"
        basic_info += str(self.nodes)
        basic_info += "\n"

        cell_n = len(self.inner)
        basic_info += f"Inner cells number: {cell_n}\n"
        for cell in self.inner:
            basic_info += cell.__str__()
            basic_info += "\n"

        cell_n = len(self.boundary)
        basic_info += f"Boundary cells number: {cell_n}\n"
        for cell in self.boundary:
            basic_info += cell.__str__()
            basic_info += "\n"

        return basic_info


# MSH 2.2 单元类型映射到维度
CELL_TYPE_DIM_MAP = {
    1: 1,  # 线段
    2: 2,  # 三角形
    3: 2,  # 四边形
    4: 3,  # 四面体
    5: 3,  # 六面体
    6: 3,  # 棱锥体
    7: 3  # 五面体
}

# 单元节点数量映射
CELL_NODE_COUNT = {
    1: 2,  # 线段
    2: 3,  # 三角形
    3: 4,  # 四边形
    4: 4,  # 四面体
    5: 8,  # 六面体
    6: 5,  # 棱锥体
    7: 6,  # 五面体
    8: 3,  # 三角形（二次）
    9: 6,  # 四边形（二次）
    10: 9,  # 四边形（二次）
    11: 10,  # 四面体（二次）
    12: 27,  # 六面体（二次）
    13: 13,  # 棱锥体（二次）
    14: 15,  # 五面体（二次）
    15: 1,  # 点单元
}


# 状态定义
class State:
    INIT = "init"
    MESH_FORMAT = "mesh_format"
    NODES_HEADER = "nodes_header"
    NODES_DATA = "nodes_data"
    ELEMENTS_HEADER = "elements_header"
    ELEMENTS_DATA = "elements_data"
    END = "end"


# 状态处理函数
def handle_init(line: str, temp: Dict, mesh: Mesh) -> str:
    if line.startswith('$'):
        section = line[1:]
        if section == "MeshFormat":
            return State.MESH_FORMAT
        elif section == "Nodes":
            return State.NODES_HEADER
        elif section == "Elements":
            return State.ELEMENTS_HEADER
        elif section == "End":
            return State.END
    return State.INIT


def handle_mesh_format(line: str, temp: Dict, mesh: Mesh) -> str:
    version = float(line.split()[0])
    if not (2.2 <= version < 4.0):
        raise ValueError(f"不支持的MSH版本: {version}（仅支持2.2-3.x）")
    return State.INIT


def handle_nodes_header(line: str, temp: Dict, mesh: Mesh) -> str:
    temp["n_nodes"] = int(line)
    temp["nodes_list"] = []
    return State.NODES_DATA


def handle_nodes_data(line: str, temp: Dict, mesh: Mesh) -> str:
    parts = list(map(float, line.split()))
    if len(parts) < 4:
        raise ValueError(f"节点格式错误: {line}（应为ID x y z）")
    temp["nodes_list"].append(parts[1:4])
    temp["n_nodes"] -= 1
    if temp["n_nodes"] == 0:
        mesh.nodes = torch.tensor(temp["nodes_list"], dtype=torch.float64)
        del temp["nodes_list"]
        return State.INIT
    return State.NODES_DATA


def handle_elements_header(line: str, temp: Dict, mesh: Mesh) -> str:
    temp["n_elements"] = int(line)
    return State.ELEMENTS_DATA


def handle_elements_data(line: str, temp: Dict, mesh: Mesh) -> str:
    parts = list(map(int, line.split()))
    if len(parts) < 4:
        raise ValueError(f"元素格式错误: {line}（至少需包含ID、type、tag_count、tags）")
    elem_id, elem_type, tag_count = parts[0], parts[1], parts[2]
    phys_id = parts[3] if tag_count >= 1 else -1
    node_indices = [idx - 1 for idx in parts[3 + tag_count:]]
    expected_count = CELL_NODE_COUNT.get(elem_type, 0)
    if len(node_indices) != expected_count:
        raise ValueError(
            f"单元类型 {elem_type} 节点数不匹配：实际{len(node_indices)}个，预期{expected_count}个"
        )
    cell = Cell(phy_id=phys_id,
                elem_type=elem_type,
                node_ids=torch.tensor(node_indices, dtype=torch.int64))
    if (is_boundary(phys_id)):
        mesh.boundary.append(cell)
    else:
        mesh.inner.append(cell)
    temp["n_elements"] -= 1
    return State.INIT if temp["n_elements"] == 0 else State.ELEMENTS_DATA


def handle_end(line: str, temp: Dict, mesh: Mesh) -> str:
    return State.END


# 状态-函数映射字典
STATE_HANDLERS: Dict[str, Callable] = {
    State.INIT: handle_init,
    State.MESH_FORMAT: handle_mesh_format,
    State.NODES_HEADER: handle_nodes_header,
    State.NODES_DATA: handle_nodes_data,
    State.ELEMENTS_HEADER: handle_elements_header,
    State.ELEMENTS_DATA: handle_elements_data,
    State.END: handle_end,
}


def parse_msh(filename: str) -> Mesh:
    mesh = Mesh()
    temp: Dict = {
        "n_nodes": 0,
        "n_elements": 0,
    }
    current_state = State.INIT

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                current_state = STATE_HANDLERS[current_state](line, temp, mesh)
            except Exception as e:
                raise type(e)(f"解析失败（行号 {line_num}）：{str(e)}") from e

            if current_state == State.END:
                break

    if mesh.inner:
        first_elem_type = mesh.inner[0].elem_type
        mesh.dim = CELL_TYPE_DIM_MAP.get(first_elem_type, 0)

    return mesh


class ElementGroupProperty:
    def __init__(self, elem_type: int, order: int, elem_ids: torch.Tensor,
                 interp_inteface_matrix: torch.Tensor,
                 deri_matrix: torch.Tensor, elem_sizes: torch.Tensor,
                 elem_to_interface: torch.Tensor):
        self.elem_type: int = elem_type
        self.elem_ids: torch.Tensor = elem_ids
        self.interp_inteface_matrix: torch.Tensor = interp_inteface_matrix.transpose(
            1, 0)
        self.deri_matrix: torch.Tensor = deri_matrix.transpose(1, 0)
        self.elem_sizes: torch.Tensor = elem_sizes
        self.elem_to_interface: torch.Tensor = elem_to_interface

    def __str__(self) -> str:
        ret = f"ElementGroupProperty:\n"
        ret += f"elem_type: {self.elem_type}\n"
        ret += f"elem_ids: {self.elem_ids.shape[0]}\n"
        ret += str(self.elem_ids)
        ret += "\ninterp_inteface_matrix: \n"
        ret += str(self.interp_inteface_matrix)
        ret += "\nderi_matrix: \n"
        ret += str(self.deri_matrix)
        ret += f"\nelem_sizes: {self.elem_sizes.shape[0]}\n"
        ret += str(self.elem_sizes)
        ret += "\nelem_to_interface: \n"
        ret += str(self.elem_to_interface)
        return ret


class BoundaryGroupProperty:
    def __init__(self, elem_type: int, phy_id: int, elem_ids: torch.Tensor,
                 boundary_to_interface: torch.Tensor):
        self.elem_type: int = elem_type
        self.phy_id: int = phy_id
        self.elem_ids: torch.Tensor = elem_ids
        self.boundary_to_interface: torch.Tensor = boundary_to_interface

    def __str__(self) -> str:
        ret = f"BoundaryGroupProperty:\n"
        ret += f"elem_type: {self.elem_type}\n"
        ret += f"phy_id: {self.phy_id}\n"
        ret += f"elem_ids: {self.elem_ids.shape[0]}\n"
        ret += str(self.elem_ids)
        ret += f"\nboundary_to_interface: {self.boundary_to_interface.shape[0]}\n"
        ret += str(self.boundary_to_interface)
        return ret


class InterfaceGroupProperty:
    def __init__(self, elem_type: int, elem_ids: torch.Tensor):
        self.elem_type: int = elem_type
        self.elem_ids: torch.Tensor = elem_ids

    def __str__(self) -> str:
        ret = f"InterfaceGroupProperty:\n"
        ret += f"elem_type: {self.elem_type}\n"
        ret += f"elem_ids: {self.elem_ids.shape[0]}\n"
        ret += str(self.elem_ids)
        return ret


class ElementGroup:
    # 扩展为2变量（h, u）
    def __init__(self, prop: ElementGroupProperty, order: int, dim_u: int):
        self.u: torch.Tensor = torch.empty(
            (prop.elem_ids.shape[0], dim_u, order + 1),
            dtype=torch.float64)  # [单元数, 变量数, 节点数]
        self.flux_deri: torch.Tensor = torch.empty_like(self.u)

    def __str__(self) -> str:
        ret = "u:\n"
        ret += str(self.u)
        ret += "\nflux_deri:\n"
        ret += str(self.flux_deri)
        return ret


class InterfaceGroup:
    # 扩展为2变量
    def __init__(self, prop: InterfaceGroupProperty, order: int, dim_u: int):
        self.u: torch.Tensor = torch.empty(
            (prop.elem_ids.shape[0], dim_u, 2),
            dtype=torch.float64)  # [界面数, 变量数, 左右]
        self.flux: torch.Tensor = torch.empty_like(self.u)
        self.restruct_flux: torch.Tensor = torch.empty(
            (prop.elem_ids.shape[0], dim_u), dtype=torch.float64)  # [界面数, 变量数]

    def __str__(self) -> str:
        ret = "InterfaceGroup:\n"
        ret += f"u:\n{self.u.shape}\n"
        ret += str(self.u)
        ret += f"\nflux:\n{self.flux.shape}\n"
        ret += str(self.flux)
        ret += f"\nrestruct_flux:\n{self.restruct_flux.shape}\n"
        ret += str(self.restruct_flux)
        return ret


def compute_interp_interface_matrix(cell: Cell, order: int) -> torch.Tensor:
    elem_type = cell.elem_type
    assert elem_type == 1
    n = order + 1
    x_nodes, _ = roots_legendre(n)
    x_targets = np.array([-1.0, 1.0], dtype=np.float64)
    interpolator = BarycentricInterpolator(x_nodes)
    interp_matrix = np.zeros((2, n))
    for j in range(n):
        f_vals = np.zeros(n)
        f_vals[j] = 1.0
        interpolator.set_yi(f_vals)
        interp_matrix[0, j] = interpolator(-1)
        interp_matrix[1, j] = interpolator(1)
    return torch.tensor(interp_matrix)


def compute_deri_matrix(cell: Cell, order: int) -> torch.Tensor:
    n = order + 1
    x, _ = roots_legendre(n)
    diff = x[:, np.newaxis] - x
    np.fill_diagonal(diff, 1)
    denom = np.prod(diff, axis=1)

    inv_diff = 1 / diff
    np.fill_diagonal(inv_diff, 0)
    diag = inv_diff.sum(axis=1)
    k, i = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    A = denom[k] / ((x[k] - x[i]) * denom[i])
    np.fill_diagonal(A, diag)
    return torch.tensor(A)


def get_mesh_groups(
    mesh_data: Mesh, order: int
) -> Tuple[List[ElementGroupProperty], List[BoundaryGroupProperty]]:
    temp: Dict = {}
    temp["elem_ids"] = []
    temp["boundary_ids"] = []
    temp["sizes"] = []
    for idx, cell in enumerate(mesh_data.inner):
        temp["elem_ids"].append(idx)
        [n0, n1] = mesh_data.inner[idx].node_ids
        temp["sizes"].append(
            abs(mesh_data.nodes[n0][0] - mesh_data.nodes[n1][0]))
    for idx, cell in enumerate(mesh_data.boundary):
        temp["boundary_ids"].append(idx)
    first_elem = mesh_data.inner[temp["elem_ids"][0]]
    interp_inteface_matrix = compute_interp_interface_matrix(first_elem, order)
    deri_matrix = compute_deri_matrix(first_elem, order)

    elem_to_interface, boundary_to_interface = get_interface_map(mesh_data)
    elem_ids = torch.tensor(temp["elem_ids"], dtype=torch.int64)
    elem_sizes = torch.tensor(temp["sizes"], dtype=torch.float64)
    elem_property = ElementGroupProperty(first_elem.elem_type, order, elem_ids,
                                         interp_inteface_matrix, deri_matrix,
                                         elem_sizes, elem_to_interface)
    first_elem = mesh_data.boundary[temp["boundary_ids"]
                                    [0]] if temp["boundary_ids"] else None
    boundary_property = BoundaryGroupProperty(
        first_elem.elem_type if first_elem else 1, 101,
        torch.tensor(temp["boundary_ids"], dtype=torch.int64),
        boundary_to_interface)

    return [elem_property], [boundary_property]


def get_interface_property(mesh_data: Mesh) -> List[InterfaceGroupProperty]:
    nodes_num = mesh_data.nodes.shape[0]
    return [InterfaceGroupProperty(15, torch.arange(0, nodes_num))]


def get_interface_map(mesh_data: Mesh):
    elem_to_interface: torch.Tensor = torch.full((len(mesh_data.inner), 2, 2),
                                                 -1,
                                                 dtype=torch.int64)
    boundary_to_interface: torch.Tensor = torch.full(
        (len(mesh_data.boundary), 2), -1, dtype=torch.int64)
    tmp: Dict = {}
    for idx, cell in enumerate(mesh_data.boundary):
        tmp[cell.node_ids[0].item()] = idx

    for idx, cell in enumerate(mesh_data.inner):
        node0 = cell.node_ids[0].item()
        node1 = cell.node_ids[1].item()
        p0 = mesh_data.nodes[node0]
        p1 = mesh_data.nodes[node1]
        if (p0[0] < p1[0]):
            elem_to_interface[idx, 0, :] = torch.tensor([node0, 1])
            if (node0 in tmp):
                boundary_to_interface[tmp[node0]] = torch.tensor([node0, 0])
            elem_to_interface[idx, 1, :] = torch.tensor([node1, 0])
            if (node1 in tmp):
                boundary_to_interface[tmp[node1]] = torch.tensor([node1, 1])
        else:
            elem_to_interface[idx, 0, :] = torch.tensor([node0, 0])
            if (node0 in tmp):
                boundary_to_interface[tmp[node0]] = torch.tensor([node0, 1])
                elem_to_interface[idx, 1, :] = torch.tensor([node1, 1])
            if (node1 in tmp):
                boundary_to_interface[tmp[node1]] = torch.tensor([node1, 0])

    return elem_to_interface, boundary_to_interface
