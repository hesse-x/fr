"""Portable CSV and SVG output without a NumPy dependency."""
import csv
from .equations import Euler


def export(path, solver, U):
    equation = getattr(solver, 'equation', Euler(2))
    rho,velocity,p = equation.primitive(U)
    u,v = velocity[...,0],velocity[...,1]
    mach = (u*u+v*v).sqrt()/(equation.gamma*p/rho).sqrt()
    err = (rho-solver.exact[...,0]).abs()
    arrays = [solver.x,solver.y,rho,u,v,p,mach,err]
    with path.with_suffix('.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['x','y','rho','u','v','pressure','Mach','abs_density_error'])
        writer.writerows(zip(*(a.flatten().tolist() for a in arrays)))
    # Solution-point tiles; these are samples, not contour interpolation.
    bounds_label = "; ".join(f"{axis}: [{lo:.4g}, {hi:.4g}]" for axis,(lo,hi) in zip("xy",solver.bounds))
    nx,ny = (c*(solver.p+1) for c in solver.cells)
    order=(0,2,1,3)
    fields=[('Density',rho),('Mach',mach),('Absolute density error',err)]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="435" viewBox="0 0 1080 435">', '<rect width="1080" height="435" fill="white"/>']
    for panel,(label,field) in enumerate(fields):
        vals=field.permute(order).reshape(nx,ny).tolist()
        low,high=field.min().item(),field.max().item()
        ox=panel*360+25
        parts.append(f'<text x="{ox}" y="25" font-family="sans-serif" font-size="16">{label}</text>')
        for i in range(nx):
            for j in range(ny):
                t=(vals[i][j]-low)/max(high-low,1e-30)
                color=f'rgb({int(255*t)},60,{int(255*(1-t))})'
                parts.append(f'<rect x="{ox+300*i/nx:.3f}" y="{45+300*(ny-j-1)/ny:.3f}" width="{300/nx+.1:.3f}" height="{300/ny+.1:.3f}" fill="{color}"/>')
        parts.append(f'<text x="{ox}" y="370" font-family="sans-serif" font-size="12">blue {low:.5g} — red {high:.5g}</text>')
        parts.append(f'<text x="{ox}" y="395" font-family="sans-serif" font-size="12">{bounds_label}; Gauss samples</text>')
    parts.append('</svg>')
    path.with_suffix('.svg').write_text('\n'.join(parts))


def export_cloud(path,solver,U,t):
    from .plotting import PlotRecorder,euler_fields
    PlotRecorder(solver,euler_fields(solver.equation),t,frames=0).finish(path,U,t)
