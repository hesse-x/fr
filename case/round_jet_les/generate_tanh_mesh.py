"""Generate the 25D SP3 cylinder with a square core and tanh radial spacing."""
import argparse
from filecmp import cmp
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

from fr.hex_geometry import _GMSH_NODES

DEFAULT_CONFIG = Path(__file__).with_name('default.json')


def load_case_config():
    with DEFAULT_CONFIG.open() as stream:
        return json.load(stream)


def axial_edges(reference):
    """Distribute 64 layers using SP3 centerline decay and a 1.03 growth cap."""
    from .compare import read_sp3

    with zipfile.ZipFile(reference) as archive:
        data = read_sp3(archive, 'SimpleX')
    center = data[(np.abs(data[:, 1]) < 1e-8) & (data[:, 3] > 0)]
    center = center[np.argsort(center[:, 0])]
    x = np.linspace(0., 25., 25001)
    u = np.interp(x, np.r_[0., center[:, 0]], np.r_[1., center[:, 2]])
    upper = np.maximum.accumulate(u[::-1])[::-1]
    cumulative = np.r_[0., np.cumsum(.5*(upper[1:]+upper[:-1])*np.diff(x))]
    target = np.diff(np.interp(np.linspace(0., cumulative[-1], 65), cumulative, x))

    def widths(scale):
        result = scale*target
        result[0] = .21
        for k in range(1, len(result)):
            result[k] = np.clip(result[k], result[k-1], 1.03*result[k-1])
        return result

    lo, hi = 0., 2.
    for _ in range(54):
        scale = (lo+hi)/2
        if widths(scale).sum() < 25.:
            lo = scale
        else:
            hi = scale
    edges = np.r_[0., np.cumsum(widths((lo+hi)/2))]
    edges[-1] = 25.
    return edges


def cross_sections(*, radius=6., circumferential_cells=24, radial_cells=5,
                   core_half_width=.6, tanh_beta=2.1):
    half_width, beta = core_half_width, tanh_beta
    if not np.isfinite(half_width) or half_width <= 0 or not np.isfinite(beta) or beta <= 0:
        raise ValueError('core half-width and tanh strength must be positive and finite')
    if not np.isfinite(radius) or radius <= np.sqrt(2.)*half_width:
        raise ValueError('cylinder radius must enclose the square core')
    core_cells, remainder = divmod(circumferential_cells, 4)
    if remainder or core_cells < 1 or radial_cells < 1:
        raise ValueError('circumferential cells must be a positive multiple of four and radial cells positive')
    side = np.linspace(-half_width, half_width, 2*core_cells+1)
    core = np.stack(np.meshgrid(side, side, indexing='ij'), axis=-1)
    # Uniform angular spacing on the circle and uniform spacing along the square.
    # Orient each outer block with clockwise tangential and outward radial axes.
    offset = side[::-1, None]
    fraction = 1.-np.tanh(beta*np.linspace(1., 0., 2*radial_cells+1))/np.tanh(beta)
    r = np.hypot(half_width, offset)*(1.-fraction)+radius*fraction
    angle = np.arctan2(offset, half_width)*(1.-fraction)
    angle += np.linspace(np.pi/4, -np.pi/4, len(side))[:, None]*fraction
    y, z = r*np.cos(angle), r*np.sin(angle)
    return [core, *(np.stack(block, axis=-1) for block in ((y,z), (-z,y), (-y,-z), (z,-y)))]


def write_mesh(output, edges, *, radius=6., circumferential_cells=24, radial_cells=5,
               core_half_width=.6, tanh_beta=2.1, replace=True):
    """Write a conforming, axially ordered Gmsh 2.2 hex27 mesh."""
    edges = np.asarray(edges, dtype=np.float64)
    if (edges.ndim != 1 or len(edges) < 2 or edges[0] != 0. or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0.)):
        raise ValueError('axial edges must increase strictly from zero')
    coordinates, point_ids, blocks = [], {}, []
    axial = np.empty(2*len(edges)-1)
    axial[::2], axial[1::2] = edges, (edges[:-1]+edges[1:])/2
    for section in cross_sections(radius=radius, circumferential_cells=circumferential_cells,
                                  radial_cells=radial_cells, core_half_width=core_half_width,
                                  tanh_beta=tanh_beta):
        points = section.reshape(-1, 2)
        ids = np.empty(len(points), dtype=np.int64)
        for j, (point, rounded) in enumerate(zip(points, np.round(points, 12))):
            index = point_ids.setdefault(tuple(rounded), len(coordinates))
            if index == len(coordinates):
                coordinates.append(point)
            ids[j] = index
        blocks.append(ids.reshape(section.shape[:2]))
    hexahedron = np.asarray(_GMSH_NODES)+1
    quad = np.array([(0,0), (2,0), (2,2), (0,2), (1,0), (2,1), (1,2), (0,1), (1,1)])
    inlet = np.column_stack((quad, np.zeros(len(quad), dtype=int)))
    outlet = np.column_stack((quad, np.full(len(quad), 2)))
    ambient = np.column_stack((quad[:, 0], np.full(len(quad), 2), quad[:, 1]))
    elements = []
    layers = len(edges)-1
    for b, block in enumerate(blocks):
        ni, nj = (np.array(block.shape)-1)//2
        for i, j, k in np.ndindex(ni, nj, layers):
            def tags(offsets):
                row, col, layer = offsets.T
                return block[2*i+row, 2*j+col]*len(axial)+2*k+layer+1

            elements.append((12, 4, tags(hexahedron)))
            if k == 0:
                elements.append((10, 1, tags(inlet)))
            if k == layers-1:
                elements.append((10, 2, tags(outlet)))
            if b and j == nj-1:
                elements.append((10, 3, tags(ambient)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        generated = Path(temporary)/output.name
        with generated.open('w') as stream:
            stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n4\n'
                         '2 1 "inlet"\n2 2 "outlet"\n2 3 "ambient"\n3 4 "fluid"\n$EndPhysicalNames\n')
            stream.write(f'$Nodes\n{len(coordinates)*len(axial)}\n')
            for index, (y, z) in enumerate(coordinates):
                for k, x in enumerate(axial):
                    stream.write(f'{index*len(axial)+k+1} {x:.17g} {y:.17g} {z:.17g}\n')
            stream.write(f'$EndNodes\n$Elements\n{len(elements)}\n')
            for i, (kind, physical, nodes) in enumerate(elements, 1):
                node_tags = ' '.join(map(str, nodes))
                stream.write(f'{i} {kind} 2 {physical} {physical} {node_tags}\n')
            stream.write('$EndElements\n')
        if not replace and output.exists():
            if not cmp(output, generated, shallow=False):
                raise FileExistsError(f'{output} differs from the case configuration; select a new --output')
        else:
            generated.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', nargs='?', type=Path, help='NASA SP3 archive to recompute axial edges')
    parser.add_argument('--radius', type=float, help='cylinder radius in nozzle diameters')
    parser.add_argument('--circumferential-cells', type=int)
    parser.add_argument('--radial-cells', type=int, help='outer-ring layers on each side')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = load_case_config()
    mesh = config['mesh'].copy()
    for key in ('radius', 'circumferential_cells', 'radial_cells'):
        if getattr(args, key) is not None:
            mesh[key] = getattr(args, key)
    output = args.output
    if output is None:
        directory = Path(config['solver']['output'])
        if mesh != config['mesh']:
            name = 'sp3_cylindrical_25d_tanh'
            if (mesh['radius'], mesh['circumferential_cells'], mesh['radial_cells']) != (6., 24, 5):
                name += '_r{radius:g}_theta{circumferential_cells}_radial{radial_cells}'.format(**mesh)
            directory = directory.with_name(name)
        output = directory/'mesh.msh'
    try:
        if args.reference is not None:
            mesh['edges'] = axial_edges(args.reference)
        write_mesh(output, **mesh, replace=args.output is not None)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(output)


if __name__ == '__main__':
    main()
