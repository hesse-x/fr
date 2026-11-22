"""Generate a round-jet cylinder with a five-block hex cross-section."""
import argparse
import math
from pathlib import Path

from fr.gmsh import generate_mesh

from .generate_mesh import build_mesh


def axial_edges(length=40., *, axial_cells=None):
    if not math.isfinite(length) or length <= 0:
        raise ValueError('cylinder length must be positive and finite')
    if axial_cells is not None:
        if axial_cells < 1:
            raise ValueError('uniform axial layer count must be positive')
        return [length*(i/axial_cells) for i in range(axial_cells+1)]
    near = build_mesh((32, 24, 24)).edges[0].tolist()
    if length < near[-1]:
        raise ValueError('stretched cylinder length must be at least 16D')
    if length == near[-1]:
        return near
    # SP3 centerline decay fit on 15–25D gives Uj/Uc proportional to x-x0.
    virtual_origin = .64
    span = (length-virtual_origin)/(near[-1]-virtual_origin)
    reference_span = (40.-virtual_origin)/(near[-1]-virtual_origin)
    downstream_layers = math.ceil(12*(math.log(span)/math.log(reference_span)))
    ratio = span**(1/downstream_layers)
    tail = [virtual_origin+(near[-1]-virtual_origin)*ratio**i for i in range(1, downstream_layers)]
    return near + tail + [length]


def geometry(length=40., *, axial_cells=None):
    radius, core_half_width = 6., .6
    circumferential_per_block, radial_cells = 16, 10
    radial_growth = 1.34
    edges = axial_edges(length, axial_cells=axial_cells)
    diagonal = radius/math.sqrt(2)
    corners = ((1, -1), (1, 1), (-1, 1), (-1, -1))
    lines = ['SetFactory("Built-in");']
    for first, scale in ((1, core_half_width), (5, diagonal)):
        for i, (y, z) in enumerate(corners, first):
            lines.append(f'Point({i}) = {{0, {scale*y:.17g}, {scale*z:.17g}}};')
    lines.append('Point(9) = {0, 0, 0};')
    for i in range(1, 5):
        following = i % 4+1
        lines.extend((f'Line({i}) = {{{i}, {following}}};',
                      f'Circle({i+4}) = {{{i+4}, 9, {following+4}}};',
                      f'Line({i+8}) = {{{i}, {i+4}}};'))
    lines.extend(('Curve Loop(1) = {1, 2, 3, 4};', 'Plane Surface(1) = {1};'))
    for i in range(1, 5):
        following = i % 4+1
        lines.extend((f'Curve Loop({i+1}) = {{{i+8}, {i+4}, -{following+8}, -{i}}};',
                      f'Plane Surface({i+1}) = {{{i+1}}};'))
    layer_counts = ', '.join('1' for _ in edges[1:])
    layer_positions = ', '.join(f'{x/edges[-1]:.17g}' for x in edges[1:])
    lines.extend((
        f'Transfinite Curve {{1:8}} = {circumferential_per_block+1};',
        f'Transfinite Curve {{9:12}} = {radial_cells+1} Using Progression {radial_growth};',
        'Transfinite Surface {1:5};',
        'Recombine Surface {1:5};',
        f'out[] = Extrude {{{edges[-1]:.17g}, 0, 0}} {{',
        '  Surface {1:5};',
        f'  Layers {{{{{layer_counts}}}, {{{layer_positions}}}}};',
        '  Recombine;',
        '};',
        'Physical Surface("inlet") = {1:5};',
        'Physical Surface("outlet") = {out[0], out[6], out[12], out[18], out[24]};',
        'Physical Surface("ambient") = {out[9], out[15], out[21], out[27]};',
        'Physical Volume("fluid") = {out[1], out[7], out[13], out[19], out[25]};',
        'Mesh.SecondOrderIncomplete = 0;',
        'Mesh.SecondOrderLinear = 0;',
    ))
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=float, default=40., help='cylinder length in nozzle diameters')
    parser.add_argument('--axial-cells', type=int, help='number of uniformly spaced axial cells')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    name = f'sp3_cylindrical_{args.length:g}d'
    if args.axial_cells is not None:
        name += f'_uniform{args.axial_cells}'
    output = args.output or Path('results/round_jet_les')/name/'mesh.msh'
    if output.suffix.lower() != '.msh':
        parser.error('output must use the .msh suffix')
    try:
        source = geometry(args.length, axial_cells=args.axial_cells)
    except ValueError as error:
        parser.error(str(error))
    geometry_path = output.with_suffix('.geo')
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text(source)
    generate_mesh(geometry_path, output, order=2, version='2.2', dimension=3)
    print(output)


if __name__ == '__main__':
    main()
