"""Generate structured O-grid quads or wall/wake-refined curved triangles."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from fr import Mesh2D, MeshFR2D
from fr.equations import Euler
from fr.gmsh import generate_mesh
from fr.unstructured import make_normal_numerical_flux


def geometry(radius=10., wall_size=.04, far_size=2., downstream=20.,
             wake_size=None, wake_length=None, wake_half_width=2.):
    wake_size=2*wall_size if wake_size is None else wake_size
    wake_length=min(14.,downstream) if wake_length is None else wake_length
    if (not all(math.isfinite(v) and v > 0 for v in
                (radius, wall_size, far_size, downstream,wake_size,wake_length,wake_half_width))
            or radius < 2 or downstream < 2 or wake_length>downstream or wake_half_width>=radius):
        raise ValueError('positive finite sizes and radius/downstream >= 2 required')
    rows = ['Point(1)={0,0,0};']
    for i, (x, y) in enumerate(((.5,0),(0,.5),(-.5,0),(0,-.5)), 2):
        rows.append(f'Point({i})={{{x},{y},0,{wall_size}}};')
    for i in range(4):
        rows.append(f'Circle({i+1})={{{i+2},1,{(i+1)%4+2}}};')
    for i, (x, y) in enumerate(((-radius,-radius),(downstream,-radius),
                                (downstream,radius),(-radius,radius)), 11):
        rows.append(f'Point({i})={{{x},{y},0,{far_size}}};')
    for i in range(4):
        rows.append(f'Line({11+i})={{{11+i},{11+(i+1)%4}}};')
    rows += ['Curve Loop(1)={1,2,3,4};', 'Curve Loop(2)={11,12,13,14};',
             'Plane Surface(1)={2,1};', 'Physical Surface("fluid",1)={1};',
             'Physical Curve("cylinder",2)={1,2,3,4};',
             'Physical Curve("farfield",1)={11,12,13,14};',
             'Field[1]=Distance;', 'Field[1].CurvesList={1,2,3,4};',
             'Field[1].Sampling=400;', 'Field[2]=Threshold;',
             'Field[2].InField=1;', f'Field[2].SizeMin={wall_size};',
             f'Field[2].SizeMax={far_size};', 'Field[2].DistMin=0.1;',
             f'Field[2].DistMax={radius/2};', 'Field[3]=Box;',
             f'Field[3].VIn={wake_size};', f'Field[3].VOut={far_size};',
             'Field[3].XMin=0.4;', f'Field[3].XMax={wake_length};',
             f'Field[3].YMin={-wake_half_width};', f'Field[3].YMax={wake_half_width};',
             'Field[3].Thickness=0.5;', 'Field[4]=Min;',
             'Field[4].FieldsList={2,3};', 'Background Field=4;',
             'Mesh.Algorithm=6;', 'Mesh.SecondOrderIncomplete=0;',
             'Mesh.HighOrderOptimize=1;']
    return '\n'.join(rows)+'\n'


def write_preview(space,output,radius,downstream,kind):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-matplotlib')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        return
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    from fr.reference import edge_points
    segments=[]
    for block in space.blocks:
        points,_=edge_points(block['basis'].kind,torch.linspace(0,1,9,dtype=torch.float64))
        for edge in points:
            segments.extend(torch.einsum('qi,eik->eqk',block['geometry'].evaluate(edge),block['X']).numpy())
    for ax in axes:
        ax.add_collection(LineCollection(segments,colors='#36566c',linewidths=.25))
        ax.set(aspect='equal',xlabel='x / D',ylabel='y / D')
    axes[0].set(xlim=(-radius,downstream),ylim=(-radius,radius),title=f'{kind}: full domain')
    axes[1].set(xlim=(-2.5,5),ylim=(-2.5,2.5),title='Cylinder and wake')
    fig.savefig(Path(output).with_suffix('.png'),dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['quad','tri'],default='quad',help='structured multiblock quads (default), or unstructured triangles')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--radius', type=float, default=10., help='upstream/top/bottom distance in diameters')
    parser.add_argument('--downstream', type=float, default=20., help='outlet x/D measured from cylinder center')
    parser.add_argument('--wall-size', type=float, default=.04,help='triangular mesh wall size in D')
    parser.add_argument('--far-size', type=float, default=2.,help='triangular mesh farfield size in D')
    parser.add_argument('--wake-size',type=float,help='triangular wake size in D (default: twice wall size)')
    parser.add_argument('--wake-length',type=float,help='triangular refined-wake end, x/D (default: min(14, downstream))')
    parser.add_argument('--wake-half-width',type=float,default=2.,help='triangular refined-wake half-width in D')
    parser.add_argument('--order', type=int, choices=range(1,6), default=3)
    parser.add_argument('--box',type=float,default=2.,help='structured core square half-width in D')
    parser.add_argument('--circumferential',type=int,default=20,help='structured cells per quarter circumference / core side')
    parser.add_argument('--radial',type=int,default=12,help='structured layers from cylinder to core square')
    parser.add_argument('--upstream-cells',type=int,default=20)
    parser.add_argument('--downstream-cells',type=int,default=80)
    parser.add_argument('--outer-cells',type=int,default=20,help='structured cells from core to top/bottom')
    parser.add_argument('--stretch',type=float,default=1.08,help='structured outer-arm spacing ratio')
    parser.add_argument('--radial-stretch',type=float,default=1.1)
    args = parser.parse_args()
    if args.output is None:
        args.output=Path(__file__).resolve().parent/('results/mesh/structured.msh' if args.kind=='quad' else 'results/mesh/coarse.msh')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if args.kind=='quad':
        from structured_mesh import geometry as structured_geometry
        geo=structured_geometry(args.radius,args.downstream,args.box,args.circumferential,
            args.radial,args.upstream_cells,args.downstream_cells,args.outer_cells,
            args.stretch,args.radial_stretch)
    else:
        geo = geometry(args.radius, args.wall_size, args.far_size, args.downstream,
                       args.wake_size,args.wake_length,args.wake_half_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.geo').write_text(geo)
    generate_mesh(args.output.with_suffix('.geo'), args.output, order=args.order)
    mesh = Mesh2D.read(args.output)
    tags = set(mesh.face_tags[mesh.boundary].tolist())
    if tags != {1,2}:
        raise RuntimeError(f'unexpected boundary tags: {tags}')
    eq = Euler(2)
    space = MeshFR2D(mesh, p=1, flux=eq.flux,
                     numerical_flux=make_normal_numerical_flux('rusanov',eq),
                     speed=eq.normal_speed, boundary=lambda coords,U,n,tag,t: U)
    space.write_svg(args.output.with_suffix('.svg'))
    write_preview(space,args.output,args.radius,args.downstream,args.kind)
    report = dict(diameter=1., kind=args.kind,geometry_order=args.order,
                  cells=sum(len(b.geometry) for b in mesh.blocks),
                  ndof_p1=space.ndof, boundary_tags=sorted(tags),
                  minimum_jacobian=float(space.minimum_jacobian), area=float(space.volume),
                  expected_area=2*args.radius*(args.radius+args.downstream)-math.pi/4,
                  bounds=[[-args.radius,args.downstream],[-args.radius,args.radius]],
                  wall_size=args.wall_size, far_size=args.far_size,
                  wake_size=2*args.wall_size if args.wake_size is None else args.wake_size,
                  wake_length=min(14.,args.downstream) if args.wake_length is None else args.wake_length,
                  wake_half_width=args.wake_half_width,
                  validation='MeshFR2D geometry, Jacobian and shared-face checks passed')
    if args.kind=='quad':
        for key in ('wall_size','far_size','wake_size','wake_length','wake_half_width'):
            report.pop(key)
        report.update(box=args.box,circumferential=args.circumferential,radial=args.radial,
            upstream_cells=args.upstream_cells,downstream_cells=args.downstream_cells,
            outer_cells=args.outer_cells,stretch=args.stretch,radial_stretch=args.radial_stretch)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
