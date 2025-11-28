import mesh
import json
import torch
import utils
import elem_map
from typing import Any, Dict, Optional


class Config:
    def __init__(self, order: int, dim_u: int, CFL: float):
        """
        Initialize a configuration object with specific parameters for numerical simulation.

        Args:
            order (int): The order of accuracy of the numerical method.
            dim_u (int): The dimension of the state vector.
            CFL (float): The CFL condition number, used to control numerical stability.
        """
        self.order = order
        self.dim_u = dim_u
        self.CFL = CFL

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Initialize a Config object from a dictionary."""
        required_keys = ["order", "dim_u", "CFL"]
        if not all(key in config_dict for key in required_keys):
            missing_keys = [
                key for key in required_keys if key not in config_dict
            ]
            raise ValueError(
                f"Configuration dictionary is missing required keys: {', '.join(missing_keys)}"
            )

        # Type checking
        if not isinstance(config_dict["order"], int):
            raise TypeError("'order' must be an integer.")
        if not isinstance(config_dict["dim_u"], int):
            raise TypeError("'dim_u' must be an integer.")
        if not isinstance(config_dict["CFL"], (int, float)):
            raise TypeError("'CFL' must be a float or integer.")

        return cls(order=config_dict["order"],
                   dim_u=config_dict["dim_u"],
                   CFL=float(config_dict["CFL"]))

    @classmethod
    def from_string(cls, config_str: str) -> "Config":
        """Initialize a Config object from a JSON string."""
        try:
            config_dict = json.loads(config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON configuration string: {e}") from e
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Config object to a dictionary."""
        return {
            "order": self.order,
            "dim_u": self.dim_u,
            "CFL": self.CFL,
        }

    def to_string(self) -> str:
        """Convert the Config object to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return (
            f"Config(order={self.order}, dim_u={self.dim_u}, CFL={self.CFL}")


class Equation:
    def __init__(self, F, roe, init, eigen_u):
        self.F = F
        self.roe = roe
        self.init = init
        self.eigen_u = eigen_u


class Boundary:
    def __init__(self, fn):
        self.fn = fn


class Solver:
    @staticmethod
    def compute_dx(prop: mesh.ElementGroupProperty, mesh_read: mesh.Mesh):
        result = torch.empty((prop.elem_ids.shape[0], ), dtype=torch.float64)
        for idx, elem_id in enumerate(prop.elem_ids):
            [n0, n1] = mesh_read.inner[elem_id].node_ids
            x0 = mesh_read.nodes[n0][0]
            x1 = mesh_read.nodes[n1][0]
            result[idx] = abs(x0 - x1)
        return result[:, None]

    def __init__(self, mesh_: mesh.Mesh, equation: Equation,
                 boundary: Boundary, correction, config: Config):
        self.mesh_ = mesh_
        self.equation = equation
        self.boundary = boundary
        self.config = config
        self.elem_prop, self.boundary_prop = mesh.get_mesh_groups(
            mesh_, self.config.order)
        self.elem_prop = self.elem_prop[0]
        self.boundary_prop = self.boundary_prop[0]
        self.interface_prop = mesh.get_interface_property(mesh_)[0]
        self.field = mesh.ElementGroup(self.elem_prop, self.config.order,
                                       self.config.dim_u)
        self.interface = mesh.InterfaceGroup(self.interface_prop,
                                             self.config.order,
                                             self.config.dim_u)
        self.dx_elem = Solver.compute_dx(self.elem_prop, self.mesh_)
        self.equation.init(self.field, self.elem_prop, self.mesh_)
        self.t = 0
        self.correction = correction

    def compute_dt(self):
        cur_u = self.equation.eigen_u(self.field.u)
        return self.config.CFL * torch.min(self.dx_elem / torch.abs(cur_u))

    def advance(self):
        dt = self.compute_dt()
        # 插值边界的u
        interped_u = utils.interp_interface(self.field.u, self.elem_prop)
        elem_map.scatter_value_to_interface(interped_u, self.interface.u,
                                            self.elem_prop)

        # 计算求解点f
        flux_p = self.equation.F(self.field.u)
        interped_flux = utils.interp_interface(flux_p, self.elem_prop)
        elem_map.scatter_value_to_interface(interped_flux, self.interface.flux,
                                            self.elem_prop)

        # 计算边界的u/f
        self.boundary.fn(self.interface, self.boundary_prop)

        # 重构通量面通量roe_f
        self.interface.restruct_flux = self.equation.roe(
            self.interface.u, self.interface.flux)

        # 计算求解点通量导数
        elem_bound_flux_diff = elem_map.gather_interface_vaue(
            self.interface.restruct_flux[:, :, None] - self.interface.flux,
            self.elem_prop, self.config.dim_u)
        self.field.flux_deri = torch.matmul(
            flux_p,
            self.elem_prop.deri_matrix) + self.correction.compute_gl_deri(
                elem_bound_flux_diff[:, :,
                                     0]) + self.correction.compute_gr_deri(
                                         elem_bound_flux_diff[:, :, 1])

        # 推进u
        #        ori_u = self.field.u
        self.field.u = self.field.u - self.field.flux_deri * dt * 2 / self.dx_elem[:, :,
                                                                                   None]
        #        diff = torch.sum(self.field.u - ori_u)
        #        print(diff)
        self.t += dt
