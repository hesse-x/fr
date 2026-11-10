"""Generate conforming cylinder or irregular meshes with the installed Gmsh CLI."""
import argparse
from pathlib import Path
from fr.gmsh import generate_mesh


def cylinder_quad_geometry(radius=.5,extent=2.5,angular=4,radial=8,interior=2):
    if not 0<radius<extent or min(angular,radial,interior)<1: raise ValueError('invalid geometry or cell counts')
    b=.4*radius
    a=radius/2**.5
    signs=((1,1),(-1,1),(-1,-1),(1,-1))
    rows=['Point(1)={0,0,0};']
    for base,scale in ((10,b),(20,a),(30,extent)):
        for i,(x,y) in enumerate(signs): rows.append(f'Point({base+i})={{{x*scale},{y*scale},0}};')
    for i in range(4):
        j=(i+1)%4
        rows.extend((f'Line({100+i})={{{10+i},{10+j}}};',
                     f'Circle({200+i})={{{20+i},1,{20+j}}};',
                     f'Line({300+i})={{{30+i},{30+j}}};',
                     f'Line({400+i})={{{10+i},{20+i}}};',
                     f'Line({500+i})={{{20+i},{30+i}}};'))
    rows.extend(('Curve Loop(1)={100,101,102,103};','Plane Surface(1)={1};','Transfinite Surface{1};'))
    for i in range(4):
        j=(i+1)%4
        rows.extend((f'Curve Loop({10+i})={{{-100-i},{400+i},{200+i},{-400-j}}};',
                     f'Plane Surface({10+i})={{{10+i}}};',
                     f'Transfinite Surface{{{10+i}}}={{{10+j},{10+i},{20+i},{20+j}}};',
                     f'Curve Loop({20+i})={{{-200-i},{500+i},{300+i},{-500-j}}};',
                     f'Plane Surface({20+i})={{{20+i}}};',
                     f'Transfinite Surface{{{20+i}}}={{{20+j},{20+i},{30+i},{30+j}}};'))
    rows.extend((f'Transfinite Curve{{100:103,200:203,300:303}}={angular+1};',
                 f'Transfinite Curve{{400:403}}={interior+1};',
                 f'Transfinite Curve{{500:503}}={radial+1};',
                 'Recombine Surface{1,10:13,20:23};','Mesh.SecondOrderIncomplete=0;',
                 'Physical Surface("dielectric",2)={1,10:13};',
                 'Physical Surface("air",1)={20:23};',
                 'Physical Curve("interface",2)={200:203};',
                 'Physical Curve("outer",1)={300:303};'))
    return '\n'.join(rows)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('cylinder-quad','cylinder-tri','irregular-quad'),default='cylinder-quad')
    p.add_argument('--order',type=int,default=3)
    p.add_argument('--version',choices=('2.2','4.1'),default='4.1')
    p.add_argument('--output',default='results/mesh/cylinder.msh')
    p.add_argument('--angular',type=int,default=4)
    p.add_argument('--radial',type=int,default=8)
    a=p.parse_args()
    target=Path(a.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    geo=target.with_suffix('.geo')
    if a.kind=='cylinder-quad': geo.write_text(cylinder_quad_geometry(angular=a.angular,radial=a.radial))
    else: geo.write_text((Path(__file__).parent/'mesh'/('cylinder.geo' if a.kind=='cylinder-tri' else 'irregular_quad.geo')).read_text())
    print(generate_mesh(geo,target,order=a.order,version=a.version))


if __name__=='__main__': main()
