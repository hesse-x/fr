"""Conforming multiblock quadrilateral O-grid with cross-shaped refinement."""
import math


def geometry(radius=10.,downstream=20.,box=2.,circumferential=20,radial=12,
             upstream_cells=20,downstream_cells=80,outer_cells=20,
             stretch=1.08,radial_stretch=1.1):
    counts=(circumferential,radial,upstream_cells,downstream_cells,outer_cells)
    if (any(not isinstance(n,int) or n<2 for n in counts)
            or not all(math.isfinite(v) for v in (radius,downstream,box,stretch,radial_stretch))
            or not .5<box<min(radius,downstream) or stretch<1 or radial_stretch<1):
        raise ValueError('positive block sizes, cell counts >=2 and stretching >=1 required')
    xs=(-radius,-box,box,downstream); ys=(-radius,-box,box,radius)
    point=lambda ix,iy:1+iy*4+ix
    horizontal=lambda ix,iy:100+iy*3+ix
    vertical=lambda ix,iy:200+iy*4+ix
    rows=[]
    for iy,y in enumerate(ys):
        for ix,x in enumerate(xs): rows.append(f'Point({point(ix,iy)})={{{x},{y},0}};')
    for iy in range(4):
        for ix,n in enumerate((upstream_cells,circumferential,downstream_cells)):
            line=horizontal(ix,iy)
            # Upstream lines run toward the cylinder; smallest cells at their end.
            ratio=1/stretch if ix==0 else 1.
            rows += [f'Line({line})={{{point(ix,iy)},{point(ix+1,iy)}}};',
                     f'Transfinite Curve{{{line}}}={n+1} Using Progression {ratio};']
    for iy,n in enumerate((outer_cells,circumferential,outer_cells)):
        for ix in range(4):
            line=vertical(ix,iy)
            ratio=1/stretch if iy==0 else stretch if iy==2 else 1.
            rows += [f'Line({line})={{{point(ix,iy)},{point(ix,iy+1)}}};',
                     f'Transfinite Curve{{{line}}}={n+1} Using Progression {ratio};']
    surfaces=[]
    for iy in range(3):
        for ix in range(3):
            if ix==iy==1: continue
            s=1+iy*3+ix; surfaces.append(s)
            edges=(horizontal(ix,iy),vertical(ix+1,iy),-horizontal(ix,iy+1),-vertical(ix,iy))
            corners=(point(ix,iy),point(ix+1,iy),point(ix+1,iy+1),point(ix,iy+1))
            rows += [f'Curve Loop({s})={{'+','.join(map(str,edges))+'};',
                     f'Plane Surface({s})={{{s}}};',
                     f'Transfinite Surface{{{s}}}={{'+','.join(map(str,corners))+'};',
                     f'Recombine Surface{{{s}}};']
    rows.append('Point(1000)={0,0,0};')
    for i in range(4):
        angle=math.pi/4+i*math.pi/2
        rows.append(f'Point({1001+i})={{{.5*math.cos(angle):.17g},{.5*math.sin(angle):.17g},0}};')
    outer=(point(2,2),point(1,2),point(1,1),point(2,1))
    outer_edges=(-horizontal(1,2),-vertical(1,1),horizontal(1,1),vertical(2,1))
    for i in range(4):
        arc=300+i; ray=400+i
        rows += [f'Circle({arc})={{{1001+i},1000,{1001+(i+1)%4}}};',
                 f'Line({ray})={{{1001+i},{outer[i]}}};',
                 f'Transfinite Curve{{{arc}}}={circumferential+1};',
                 f'Transfinite Curve{{{ray}}}={radial+1} Using Progression {radial_stretch};']
    for i in range(4):
        s=20+i; surfaces.append(s)
        edges=(-(300+i),400+i,outer_edges[i],-(400+(i+1)%4))
        corners=(1001+(i+1)%4,1001+i,outer[i],outer[(i+1)%4])
        rows += [f'Curve Loop({s})={{'+','.join(map(str,edges))+'};',
                 f'Plane Surface({s})={{{s}}};',
                 f'Transfinite Surface{{{s}}}={{'+','.join(map(str,corners))+'};',
                 f'Recombine Surface{{{s}}};']
    far=[horizontal(ix,iy) for iy in (0,3) for ix in range(3)]
    far += [vertical(ix,iy) for ix in (0,3) for iy in range(3)]
    rows += ['Physical Surface("fluid",1)={'+','.join(map(str,surfaces))+'};',
             'Physical Curve("cylinder",2)={300,301,302,303};',
             'Physical Curve("farfield",1)={'+','.join(map(str,far))+'};',
             'Mesh.SecondOrderIncomplete=0;','Mesh.HighOrderOptimize=1;']
    return '\n'.join(rows)+'\n'
