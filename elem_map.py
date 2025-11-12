import mesh
import torch


def scatter_value_to_interface(value: torch.Tensor,
                               interface_value: torch.Tensor,
                               elem_prop: mesh.ElementGroupProperty):
    x = elem_prop.elem_to_interface[:, 0, 0]
    y = elem_prop.elem_to_interface[:, 0, 1]
    interface_value[x, :, y] = value[:, :, 0]
    x = elem_prop.elem_to_interface[:, 1, 0]
    y = elem_prop.elem_to_interface[:, 1, 1]
    interface_value[x, :, y] = value[:, :, 1]


def gather_interface_vaue(interface_value: torch.Tensor,
                          elem_prop: mesh.ElementGroupProperty, dim_u: int):
    value = torch.empty([elem_prop.elem_to_interface.shape[0], dim_u, 2],
                        dtype=torch.float64)
    x = elem_prop.elem_to_interface[:, 0, 0]
    y = elem_prop.elem_to_interface[:, 0, 1]
    value[:, :, 0] = interface_value[x, :, y]
    x = elem_prop.elem_to_interface[:, 1, 0]
    y = elem_prop.elem_to_interface[:, 1, 1]
    value[:, :, 1] = interface_value[x, :, y]
    return value
