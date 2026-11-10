"""Zero-field Fig. 2 of De Raedt & Michielsen, Computers in Physics 8 (1994).

Uses this repository's FR-LDG spatial operator, with a Chebyshev exponential
for the time-independent Hamiltonian. The reference curve is digitized from the published figure.
Units: lambda=1, E=1, time=hbar/E; H=-Laplacian/(4*pi**2)+V.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from scipy import sparse, special
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from fr import CartesianFR, CartesianLDG, central


def axis(length, lo, h, p, edges=None):
    n = round(length/h) if edges is None else len(edges)-1
    if edges is None and not np.isclose(n*h, length):
        raise ValueError('h must divide both box lengths')
    grid = CartesianFR((n,), ((lo,lo+length),), p, flux=lambda u,a:u,
                       numerical_flux=central, periodic=False,
                       boundary=lambda c,u,a,s,t:-u,
                       edges=None if edges is None else (edges,))
    ldg = CartesianLDG(grid, boundary=lambda c,u,a,s,t:torch.zeros_like(u))
    size = n*(p+1)
    lap = ldg.diffusion(torch.eye(size,dtype=torch.float64).reshape(n,p+1,size)).reshape(size,size).numpy()
    x = grid.coords[0].numpy().ravel()
    weights = (grid.widths[0].numpy()[:,None]*grid.w.numpy()[None,:]/2).ravel()
    # Tensor identity uses physical nodal values; transform to Euclidean norm.
    root = np.sqrt(weights)
    kinetic = -root[:,None]*lap/root[None,:]/(4*np.pi**2)
    if np.max(abs(kinetic-kinetic.T)) > 1e-10:
        raise RuntimeError('FR-LDG operator is not weighted self-adjoint')
    kinetic[abs(kinetic)<1e-12] = 0
    return x, weights, sparse.csr_matrix(kinetic)


def native():
    source = Path(__file__).with_name('evolve.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = Path('/tmp')/f'fr-double-slit-{digest}.so'
    if not library.exists():
        subprocess.run(['g++','-O3','-std=c++17','-fopenmp','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    lib = ctypes.CDLL(str(library))
    lib.advance.argtypes = [ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*7+[ctypes.c_double]*2+[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
    lib.advance.restype = None
    return lib


def coefficients(radius, center, dt, tol):
    z = radius*dt
    order = max(8,math.ceil(z+12*z**(1/3)+20))
    k = np.arange(order+1)
    bessel = special.jv(k,z)
    # Bound the omitted absolute tail, rather than relying on one small term.
    tails = np.cumsum((2*abs(bessel))[::-1])[::-1]
    valid = np.flatnonzero(tails<tol/10)
    if len(valid): order = max(2,int(valid[0]))
    coeff = 2*(-1j)**k[:order+1]*bessel[:order+1]*np.exp(-1j*center*dt)
    coeff[0] /= 2
    return np.ascontiguousarray(coeff,dtype=np.complex128)


def propagate(z, hx, hy, potential, duration, dt, tol=1e-13, threads=2):
    if threads<1 or not z.flags.c_contiguous or z.dtype!=np.complex128:
        raise ValueError('positive thread count and contiguous complex128 state required')
    # Positive kinetic + nonnegative potential; Gershgorin gives a safe upper bound.
    upper = float(abs(hx).sum(axis=1).max()+abs(hy).sum(axis=1).max()+potential.max())
    radius = upper/2
    steps = max(1,math.ceil(duration/dt))
    step = duration/steps
    coeff = coefficients(radius,radius,step,tol/steps)
    arrays = [np.ascontiguousarray(a,dtype=np.int64 if i%3 in (0,1) else np.float64)
              for i,a in enumerate([hx.indptr,hx.indices,hx.data,hy.indptr,hy.indices,hy.data])]
    potential = np.ascontiguousarray(potential,dtype=np.float64)
    args = [a.ctypes.data for a in arrays]+[potential.ctypes.data]
    start=time.monotonic()
    native().advance(*z.shape,*args,radius,radius,coeff.ctypes.data,len(coeff)-1,steps,z.ctypes.data,threads)
    return dict(steps=steps,dt=step,order=len(coeff)-1,spectral_upper_bound=upper,
                native_seconds=time.monotonic()-start,native_threads=threads)


def uniform_field(z,x,y,wx,wy,h,p,sample_h=.1):
    # Fourier quadrature on a uniform sampling grid, with local polynomial
    # interpolation of the complex amplitude for FR.
    psi = z/np.sqrt(wx[:,None]*wy[None,:])
    def interpolate(values,nodes,weights,lo,length,dim):
        samples = lo+(np.arange(round(length/sample_h))+.5)*sample_h
        widths=weights.reshape(-1,p+1).sum(axis=1)
        edges=np.r_[lo,lo+np.cumsum(widths)]
        cells=np.clip(np.searchsorted(edges,samples,side='right')-1,0,len(widths)-1)
        index = cells[:,None]*(p+1)+np.arange(p+1)
        selected = nodes[index]
        weights = np.ones_like(selected)
        for j in range(selected.shape[1]):
            for k in range(selected.shape[1]):
                if j!=k: weights[:,j]*=(samples-selected[:,k])/(selected[:,j]-selected[:,k])
        v = np.moveaxis(values,dim,0)
        out = np.sum(v[index]*weights[:,:,None],axis=1)
        return samples,np.moveaxis(out,0,dim)
    # Domain is inferred from the quadrature nodes and their integration weights.
    lx,ly = float(wx.sum()),float(wy.sum())
    xx,psi = interpolate(psi,x,wx,-lx/2,lx,0)
    yy,psi = interpolate(psi,y,wy,-ly/2,ly,1)
    return xx,yy,psi


def angular_profile(xx,yy,psi,sample_h,angles,padding=2):
    # Eq. (2.6): integral |FT(qx,qx*tan(theta))|^2 dqx; no extra Jacobian.
    transmitted = psi*(xx[:,None]>=4)
    shape = (padding*len(xx),padding*len(yy))
    power = abs(np.fft.fftshift(np.fft.fft2(transmitted,s=shape)))**2
    qx = np.fft.fftshift(np.fft.fftfreq(shape[0],sample_h))
    qy = np.fft.fftshift(np.fft.fftfreq(shape[1],sample_h))
    interp = RegularGridInterpolator((qx,qy),power,bounds_error=False,fill_value=0.)
    # Incident energy is ~1 with narrow spread; cutoff excludes negligible
    # high-frequency quadrature noise. It is also exported for reproducibility.
    radial = np.linspace(0,2,1601)
    profile = []
    for a in np.deg2rad(angles):
        pts = np.column_stack((radial,radial*np.tan(a)))
        profile.append(trapezoid(interp(pts),radial))
    profile = np.asarray(profile)
    return profile/profile[np.argmin(abs(angles))]


def peaks(angles,profile):
    index,_ = find_peaks(profile,prominence=.015)
    index = index[(angles[index]>=0)&(angles[index]<65)]
    return [dict(angle=float(angles[i]),height=float(profile[i])) for i in index]


def run(args):
    torch.set_num_threads(1)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    x,wx,hx = axis(args.lx,-args.lx/2,args.h,args.p)
    y,wy,hy = axis(args.ly,-args.ly/2,args.h,args.p)
    ex=np.linspace(-args.lx/2,args.lx/2,len(x)//(args.p+1)+1)
    ey=np.linspace(-args.ly/2,args.ly/2,len(y)//(args.p+1)+1)
    mesh=dict(kind='uniform',cells=[len(ex)-1,len(ey)-1],complex_dofs=len(x)*len(y),
              min_width=[float(np.diff(ex).min()),float(np.diff(ey).min())],
              max_width=[float(np.diff(ex).max()),float(np.diff(ey).max())])
    print(json.dumps(dict(mesh=mesh,backend='fused C++/OpenMP',threads=args.threads)),flush=True)
    # Fig. 1: middle block height d=4; two apertures 2<|y|<3; thickness w=4.
    potential = 100.*((x[:,None]>=-1e-10)&(x[:,None]<4-1e-10)&~((abs(y[None,:])>2+1e-10)&(abs(y[None,:])<3-1e-10)))
    z = np.exp(-((x[:,None]-args.x0)**2+y[None,:]**2)/(2*6**2)+2j*np.pi*x[:,None])
    z *= np.sqrt(wx[:,None]*wy[None,:]); z /= np.linalg.norm(z)
    initial_energy = float(np.vdot(z,hx@z+(hy@z.T).T+potential*z).real)
    start = time.monotonic()
    checkpoints = sorted(set([args.time,args.time-20]))
    history=[]; previous=0
    angles=np.linspace(-80,80,1601)
    profiles=[]
    for checkpoint in checkpoints:
        if checkpoint<=0: continue
        current=previous
        stats=dict(steps=0,native_seconds=0.)
        while current<checkpoint:
            duration=min(20.,checkpoint-current)
            part=propagate(z,hx,hy,potential,duration,args.dt,threads=args.threads)
            stats['steps']+=part.pop('steps')
            stats['native_seconds']+=part.pop('native_seconds')
            stats.update(part)
            current+=duration
            print(json.dumps(dict(evolution_time=current,native_seconds=stats['native_seconds'])),flush=True)
        xx,yy,psi=uniform_field(z,x,y,wx,wy,args.h,args.p,args.sample_h)
        profile=angular_profile(xx,yy,psi,args.sample_h,angles)
        profiles.append(profile)
        density=abs(z)**2
        history.append(dict(time=checkpoint,probability=float(density.sum()),
                            transmitted_probability=float(density[x>=4].sum()),
                            screen_probability=float(density[potential>0].sum()),
                            energy=float(np.vdot(z,hx@z+(hy@z.T).T+potential*z).real),
                            positive_peaks=peaks(angles,profile),**stats))
        previous=checkpoint
        print(json.dumps(history[-1]),flush=True)
    np.savez_compressed(out/'solution.npz',x=x,y=y,wx=wx,wy=wy,psi=z/np.sqrt(wx[:,None]*wy[None,:]),
                        x_edges=ex,y_edges=ey,angles=angles,profiles=np.asarray(profiles),times=[v['time'] for v in history])
    np.savetxt(out/'angular_profile.csv',np.column_stack((angles,*profiles)),delimiter=',',
               header='angle_degrees,'+','.join(f"t{v['time']:g}" for v in history),comments='')
    report=dict(parameters=vars(args),mesh=mesh,initial_energy=initial_energy,history=history,wall_seconds=time.monotonic()-start)
    if len(profiles)>1:
        report['late_profile_relative_l2']=float(np.linalg.norm(profiles[-1]-profiles[-2])/np.linalg.norm(profiles[-1]))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    ax.vlines(ex,-args.ly/2,args.ly/2,lw=.25,color='gray')
    ax.hlines(ey,-args.lx/2,args.lx/2,lw=.25,color='gray')
    ax.set(xlabel='x / lambda',ylabel='y / lambda',title=f"{mesh['kind']} mesh: {mesh['cells']}")
    ax.set_aspect('equal');fig.tight_layout();fig.savefig(out/'mesh.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for row,profile in zip(history,profiles): ax.plot(angles,profile,label=f"FR-LDG t={row['time']:g}")
    ax.set(xlabel='Angle (degrees)',ylabel='P(theta) / P(0)',xlim=(-80,80),ylim=(0,1.15))
    ax.legend();fig.tight_layout();fig.savefig(out/'angular_profile.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.pcolormesh(xx,yy,np.log10(abs(psi.T)**2+1e-12),vmin=-9,vmax=-2,cmap='magma',shading='nearest')
    from matplotlib.patches import Rectangle
    for lo,hi in [(-args.ly/2,-3),(-2,2),(3,args.ly/2)]: ax.add_patch(Rectangle((0,lo),4,hi-lo,color='cyan',alpha=.5))
    ax.set(xlabel='x / lambda',ylabel='y / lambda',title=f'log10 probability density, t={args.time:g}')
    fig.tight_layout();fig.savefig(out/'density.png',dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h',type=float,default=.5,help='FR cell width, in lambda')
    parser.add_argument('--threads',type=int,default=2,help='native OpenMP threads, independent of Torch setup')
    parser.add_argument('--p',type=int,default=4)
    parser.add_argument('--lx',type=float,default=104.)
    parser.add_argument('--ly',type=float,default=52.)
    parser.add_argument('--x0',type=float,default=-22.)
    parser.add_argument('--time',type=float,default=160.)
    parser.add_argument('--dt',type=float,default=2.,help='Chebyshev chunk size, not explicit RK timestep')
    parser.add_argument('--sample-h',type=float,default=.1)
    parser.add_argument('--output',help='default path is derived from mesh, p and spacing')
    args=parser.parse_args()
    if args.h<=0 or args.threads<1 or args.dt<=0 or args.time<=0 or args.sample_h<=0:
        parser.error('positive spacing, chunk size and final time required')
    if args.output is None:
        tag=f'{args.h:g}'.replace('.','')
        args.output=f'results/quantum_double_slit/fr_p{args.p}_h{tag}'
    run(args)


if __name__=='__main__': main()
