"""Conforming quad blocks: NACA O-layer, cross refinement, semicircle/box exterior."""
import math


def thickness(x):
    return .6*(.2969*math.sqrt(x)-.126*x-.3516*x*x+.2843*x**3-.1036*x**4)


def geometry(radius=15., downstream=20., surface_cells=16, radial=12,
             downstream_cells=60, outer_cells=16,
             core_left=-1., core_right=2., cross_half_width=1.,
             stretch=1.08, radial_stretch=1.15):
    counts=(surface_cells,radial,downstream_cells,outer_cells)
    values=(radius,downstream,core_left,core_right,cross_half_width,stretch,radial_stretch)
    if (any(not isinstance(n,int) or n<2 for n in counts)
            or not all(math.isfinite(v) for v in values)
            or radius<3 or downstream<3 or not .5-radius/math.sqrt(2)<core_left<0
            or not 1<core_right<1+downstream or not .1<cross_half_width<radius/math.sqrt(2)
            or stretch<1 or radial_stretch<1):
        raise ValueError('invalid C-domain block dimensions, counts, or stretching')
    rows=[]; points={}; curves={}; surfaces=[]
    def point(x,y):
        key=(round(x,14),round(y,14))
        if key not in points:
            pid=len(points)+1; points[key]=pid
            rows.append(f'Point({pid})={{{x:.17g},{y:.17g},0}};')
        return points[key]
    center=point(.5,0)
    def curve(a,b,n,ratio=1.,arc=False,spline=None):
        if (a,b) in curves: return curves[a,b]
        if (b,a) in curves: return -curves[b,a]
        cid=len(curves)+1
        curves[a,b]=cid
        if spline is not None:
            rows.append(f'Spline({cid})={{'+','.join(map(str,spline))+'};')
        elif arc:
            rows.append(f'Circle({cid})={{{a},{center},{b}}};')
        else:
            rows.append(f'Line({cid})={{{a},{b}}};')
        rows.append(f'Transfinite Curve{{{cid}}}={n+1} Using Progression {ratio:.17g};')
        return cid
    def surface(vertices,edges):
        sid=len(surfaces)+1; surfaces.append(sid)
        rows.extend([f'Curve Loop({sid})={{'+','.join(map(str,edges))+'};',
                     f'Plane Surface({sid})={{{sid}}};',
                     f'Transfinite Surface{{{sid}}}={{'+','.join(map(str,vertices))+'};',
                     f'Recombine Surface{{{sid}}};'])
    # Left semicircular annulus: every circular boundary vertex has an
    # inward grid line, avoiding zero-angle transfinite corners on a smooth arc.
    c=.5; h=cross_half_width; xl=core_left; xr=core_right; outlet=1+downstream
    outer=[point(x,y) for x,y in ((xr,0),(xr,h),(c,h),(xl,h),
                                 (xl,0),(xl,-h),(c,-h),(xr,-h))]
    far_ring={2:point(c,radius),3:point(c-radius/math.sqrt(2),radius/math.sqrt(2)),
              4:point(c-radius,0),5:point(c-radius/math.sqrt(2),-radius/math.sqrt(2)),
              6:point(c,-radius)}
    far=[]
    for i in range(2,6):
        j=i+1
        # Same exterior layer count on all shared rays and vertical arms.
        ci=curve(outer[i],far_ring[i],outer_cells,stretch)
        cj=curve(outer[j],far_ring[j],outer_cells,stretch)
        edge=curve(far_ring[i],far_ring[j],surface_cells,arc=True)
        core=curve(outer[i],outer[j],surface_cells)
        surface((outer[i],far_ring[i],far_ring[j],outer[j]),(ci,edge,-cj,-core))
        far.append(edge)
    # Top/bottom vertical refined arms, joined to rectangular downstream blocks.
    grid={}
    for ix,x in enumerate((c,xr,outlet)):
        for iy,y in enumerate((-radius,-h,0.,h,radius)):
            grid[ix,iy]=point(x,y)
    horizontal={};vertical={}
    for iy in (0,1,3,4):
        for ix,n in enumerate((surface_cells,downstream_cells)):
            edge=curve(grid[ix,iy],grid[ix+1,iy],n,1. if ix==0 else stretch)
            horizontal[ix,iy]=edge
            if iy in (0,4):far.append(edge)
    horizontal[1,2]=curve(grid[1,2],grid[2,2],downstream_cells,stretch)
    for ix in range(3):
        for iy,n in enumerate((outer_cells,surface_cells,surface_cells,outer_cells)):
            if ix==0 and iy in (1,2):continue
            ratio=1/stretch if iy==0 else stretch if iy==3 else 1.
            edge=curve(grid[ix,iy],grid[ix,iy+1],n,ratio)
            vertical[ix,iy]=edge
            if ix==2:far.append(edge)
    for ix in range(2):
        for iy in range(4):
            if ix==0 and iy in (1,2):continue
            surface((grid[ix,iy],grid[ix+1,iy],grid[ix+1,iy+1],grid[ix,iy+1]),
                    (horizontal[ix,iy],vertical[ix+1,iy],-horizontal[ix,iy+1],-vertical[ix,iy]))
    inner=[]; wall=[]
    for i in range(8):
        nodes=[]
        for j in range(21):
            theta=(i+j/20)*math.pi/4
            x=(1+math.cos(theta))/2
            y=math.copysign(thickness(x),math.sin(theta)) if x not in (0.,1.) else 0.
            nodes.append(point(x,y))
        inner.append(nodes[0])
        wall.append(curve(nodes[0],nodes[-1],surface_cells,spline=nodes))
    rays=[curve(inner[i],outer[i],radial,radial_stretch) for i in range(8)]
    for i in range(8):
        j=(i+1)%8
        outer_edge=curve(outer[i],outer[j],surface_cells)
        surface((inner[i],outer[i],outer[j],inner[j]),(rays[i],outer_edge,-rays[j],-wall[i]))
    rows.extend(['Physical Surface("fluid",1)={'+','.join(map(str,surfaces))+'};',
                 'Physical Curve("airfoil",2)={'+','.join(map(str,wall))+'};',
                 'Physical Curve("farfield",1)={'+','.join(map(str,far))+'};',
                 'Mesh.SecondOrderIncomplete=0;', 'Mesh.HighOrderOptimize=1;'])
    return '\n'.join(rows)+'\n'
