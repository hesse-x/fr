"""Generate the conforming NACA semicircle/rectangle structured quadrilateral mesh."""
import argparse
import json
import math
from pathlib import Path
import sys
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fr.gmsh import generate_mesh
from fr import Mesh2D, MeshFR2D
from fr.equations import Euler
from fr.unstructured import make_normal_numerical_flux
import torch


spec=importlib.util.spec_from_file_location('naca_structured_geometry',Path(__file__).with_name('structured_mesh.py'))
structured=importlib.util.module_from_spec(spec); spec.loader.exec_module(structured)
geometry=structured.geometry


def main(*, default_radius=15., default_output=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--radius',type=float,default=default_radius,help='semicircle radius about x/c=.5')
    p.add_argument('--downstream',type=float,default=20.,
                   help='distance from trailing edge to outlet in chords (default: 20)')
    p.add_argument('--order',type=int,choices=range(1,6),default=3)
    for name,default in (('surface-cells',16),('radial',12),
                         ('downstream-cells',60),('outer-cells',16)):
        p.add_argument('--'+name,type=int,default=default)
    for name,default in (('core-left',-1.),('core-right',2.),('cross-half-width',1.),
                         ('stretch',1.08),('radial-stretch',1.15)):
        p.add_argument('--'+name,type=float,default=default)
    a=p.parse_args()
    if a.output is None:
        a.output=default_output or Path(__file__).parent/'results/mesh/structured.msh'
    torch.set_num_threads(1)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    geo=a.output.with_suffix('.geo')
    downstream=a.radius if a.downstream is None else a.downstream
    mesh_geometry=geometry(a.radius,downstream,**{key:getattr(a,key) for key in
        ('surface_cells','radial','downstream_cells','outer_cells',
         'core_left','core_right','cross_half_width','stretch','radial_stretch')})
    geo.write_text(mesh_geometry)
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
        from fr.reference import edge_points
        segments=[]
        for block in s.blocks:
            fp,_=edge_points(block['basis'].kind,torch.linspace(0,1,13,dtype=torch.float64))
            for edge in fp:
                segments.extend(torch.einsum('qi,eik->eqk',block['geometry'].evaluate(edge),block['X']).numpy())
        fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
        for ax in axes:
            ax.add_collection(LineCollection(segments,colors='#526879',linewidths=.3))
            ax.set_aspect('equal')
            ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
        axes[0].set_xlim(.5-a.radius,downstream+1); axes[0].set_ylim(-a.radius,a.radius)
        axes[0].set_title('Semicircle + rectangle')
        axes[1].set_xlim(-3,5); axes[1].set_ylim(-3,3)
        axes[1].set_title('Cross refinement and wake')
        axes[2].set_xlim(-.15,1.2); axes[2].set_ylim(-.4,.4)
        axes[2].set_title('Body-fitted quadrilaterals')
        fig.savefig(a.output.with_suffix('.png'),dpi=180)
        plt.close(fig)
    except ImportError:
        print('Matplotlib unavailable; SVG preview is still available')
    report=dict(kind='quad',geometry_order=a.order,ndof_p1=s.ndof,boundary_tags=sorted(tags),
                cells=sum(len(b.geometry) for b in mesh.blocks),
                minimum_jacobian=float(s.minimum_jacobian),
                area=float(s.volume),radius=a.radius,
                downstream=downstream,bounds=[[.5-a.radius,downstream+1],[-a.radius,a.radius]],
                validation='MeshFR2D geometry, Jacobian and shared-face checks passed')
    report.update({key:getattr(a,key) for key in ('surface_cells','radial',
        'downstream_cells','outer_cells','core_left','core_right','cross_half_width','stretch','radial_stretch')})
    foil_area=1.2*(.2969*2/3-.126/2-.3516/3+.2843/4-.1036/5)
    report['expected_area']=math.pi*a.radius**2/2+2*a.radius*(downstream+.5)-foil_area
    report['area_relative_error']=abs(report['area']/report['expected_area']-1)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
