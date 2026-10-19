"""Composable exterior-state callbacks, with positive-axis flux convention."""

def extrapolate(coords, inside, axis, side, t):
    return inside


def prescribed(state):
    return lambda coords, inside, axis, side, t: state(coords, t)


def per_face(callbacks, default=extrapolate):
    return lambda coords, inside, axis, side, t: callbacks.get((axis, side), default)(coords, inside, axis, side, t)


def acoustic_wall(coords, inside, axis, side, t):
    ghost = inside.clone()
    ghost[..., axis+1] *= -1
    return ghost


def normal_extrapolate(coords,inside,normal,tag,t):
    return inside


def per_tag(callbacks,default=normal_extrapolate):
    """Dispatch an unstructured-mesh boundary by its integer physical tag."""
    return lambda coords,inside,normal,tag,t:callbacks.get(tag,default)(coords,inside,normal,tag,t)
