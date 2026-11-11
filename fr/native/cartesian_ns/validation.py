"""Reject unsupported model configurations before native plans are built."""
from ...equations.navier_stokes import NavierStokes


def validate_model(model):
    s = model.s
    if s.dim != 3 or s.p != 3 or s.z.device.type != 'cpu':
        raise ValueError('native-p3 requires three-dimensional CPU p=3 Cartesian NS')
    if not isinstance(model.eq, NavierStokes) or model.eq.dim != 3:
        raise ValueError('native-p3 requires the constant-property NavierStokes model')
    expected = {'periodic': (True, True, True), 'isothermal-channel': (True, False, True)}
    if expected.get(getattr(model, 'boundary_preset', None)) != s.periodic:
        raise ValueError('native-p3 supports periodic or stationary isothermal y-wall presets')
    if s.anti_alias:
        raise ValueError('native-p3 does not support flux dealiasing; select torch')
