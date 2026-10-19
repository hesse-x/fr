"""Generate a closed-trailing-edge NACA 0012 curved triangular mesh."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fr.gmsh import generate_mesh
from fr import Mesh2D, MeshFR2D
from fr.equations import Euler
from fr.unstructured import make_normal_numerical_flux
import torch


def geometry(radius=15., wall_size=.025, far_size=2., downstream=20.):
    downstream=radius if downstream is None else downstream
    if not all(math.isfinite(v) and v > 0 for v in (radius, wall_size, far_size, downstream)) or radius < 2 or downstream < 2:
        raise ValueError('positive finite sizes and radius >= 2 required')
    n = 80
    upper = []
    for i in range(n+1):
        x = .5*(1-math.cos(math.pi*i/n))
        y = .6*(.2969*math.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)
        upper.append((x, y if i not in (0,n) else 0.))
    points = upper + [(x,-y) for x,y in upper[-2:0:-1]]
    rows = [f'Point({i+1})={{{x:.17g},{y:.17g},0,{wall_size}}};' for i,(x,y) in enumerate(points)]
    rows += ['Spline(1)={'+','.join(map(str,range(1,n+2)))+'};',
             'Spline(2)={'+','.join(map(str,list(range(n+1,2*n+1))+[1]))+'};']
    for i,(x,y) in enumerate(((-radius,-radius),(downstream+1,-radius),(downstream+1,radius),(-radius,radius))):
        rows.append(f'Point({1001+i})={{{x},{y},0,{far_size}}};')
    for i in range(4):
        rows.append(f'Line({11+i})={{{1001+i},{1001+(i+1)%4}}};')
    rows += ['Curve Loop(1)={1,2};','Curve Loop(2)={11,12,13,14};',
             'Plane Surface(1)={2,1};','Physical Surface("fluid",1)={1};',
             'Physical Curve("airfoil",2)={1,2};','Physical Curve("farfield",1)={11,12,13,14};',
             'Field[1]=Distance;','Field[1].CurvesList={1,2};','Field[1].Sampling=400;',
             'Field[2]=Threshold;','Field[2].InField=1;',f'Field[2].SizeMin={wall_size};',
             f'Field[2].SizeMax={far_size};','Field[2].DistMin=0.1;',f'Field[2].DistMax={radius/2};',
             'Background Field=2;','Mesh.Algorithm=6;','Mesh.SecondOrderIncomplete=0;',
             'Mesh.HighOrderOptimize=1;']
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results/mesh/naca0012.msh')
    p.add_argument('--radius',type=float,default=15.,help='upstream/top/bottom distance in chords')
    p.add_argument('--downstream',type=float,default=20.,
                   help='distance from trailing edge to outlet in chords (default: 20)')
    p.add_argument('--wall-size',type=float,default=.025)
    p.add_argument('--far-size',type=float,default=2.)
    p.add_argument('--order',type=int,choices=range(1,6),default=3)
    a=p.parse_args()
    torch.set_num_threads(1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    geo=a.output.with_suffix('.geo')
    downstream=a.radius if a.downstream is None else a.downstream
    geo.write_text(geometry(a.radius,a.wall_size,a.far_size,downstream))
    generate_mesh(geo,a.output,order=a.order)
    mesh=Mesh2D.read(a.output)
    tags=set(mesh.face_tags[mesh.boundary].tolist())
    if tags != {1,2}: raise RuntimeError(f'unexpected boundary tags: {tags}')
    e=Euler(2)
    s=MeshFR2D(mesh,p=1,flux=e.flux,numerical_flux=make_normal_numerical_flux('rusanov',e),
               speed=e.normal_speed,boundary=lambda coords,U,n,tag,t: U)
    s.write_svg(a.output.with_suffix('.svg'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        segments=s.face_coordinates.tolist()
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax in axes:
            ax.add_collection(LineCollection(segments,colors='#526879',linewidths=.3))
            ax.set_aspect('equal')
            ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
        axes[0].set_xlim(-a.radius,downstream+1); axes[0].set_ylim(-a.radius,a.radius)
        axes[0].set_title('Full domain')
        axes[1].set_xlim(-.15,1.2); axes[1].set_ylim(-.4,.4)
        axes[1].set_title('NACA 0012: near-field mesh')
        fig.tight_layout(); fig.savefig(a.output.with_suffix('.png'),dpi=180)
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; SVG preview is still available')
    report=dict(geometry_order=a.order,ndof_p1=s.ndof,boundary_tags=sorted(tags),
                cells=sum(len(b.geometry) for b in mesh.blocks),
                minimum_jacobian=float(s.minimum_jacobian),
                area=float(s.volume),radius=a.radius,wall_size=a.wall_size,far_size=a.far_size,
                downstream=downstream,bounds=[[-a.radius,downstream+1],[-a.radius,a.radius]],
                validation='MeshFR2D geometry, Jacobian and shared-face checks passed')
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
