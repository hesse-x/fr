"""Independent double-slit accuracy study (same finite-height screen and box).

FR uses the existing LDG operator. The independent reference uses cell-centred
second-order finite differences with odd ghost values at Dirichlet walls.
Both kinetic matrices are diagonalized once; a rank-factored exact screen kick
makes fourth-order symmetric splitting economical. This is a validation driver,
not a replacement for the production RK4 solver. NumPy/Matplotlib are needed
only to export comparison figures.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import time
import torch
from fr import CartesianFR, CartesianLDG, central
from case.schrodinger import gaussian

VERSION = 1


def basis(kind, n, length, lo, p=3):
    if kind == 'fr':
        s = CartesianFR((n,), ((lo,lo+length),), p, flux=lambda U,a: U,
                        numerical_flux=central, periodic=False,
                        boundary=lambda c,U,a,side,t: U*0)
        op = CartesianLDG(s,lambda c,U,a,side,t: U*0)
        size = n*(p+1)
        identity = torch.eye(size,dtype=torch.float64).reshape(n,p+1,size)
        lap = op.diffusion(identity).reshape(size,size)
        weights = s.w.repeat(n)*length/(2*n)
        root = weights.sqrt()
        symmetric = root[:,None]*lap/root[None,:]
        if not torch.allclose(symmetric,symmetric.T,atol=1e-10,rtol=1e-12):
            raise RuntimeError('LDG Laplacian lost weighted self-adjointness')
        eig, Q = torch.linalg.eigh((symmetric+symmetric.T)/2)
        return s.coords[0].reshape(-1), weights, -.5*eig, Q
    if kind != 'fd':
        raise ValueError('kind must be fr or fd')
    h = length/n
    j = torch.arange(n,dtype=torch.float64)
    k = torch.arange(1,n+1,dtype=torch.float64)
    Q = (2/n)**.5*torch.sin(math.pi*(j[:,None]+.5)*k[None,:]/n)
    Q[:,-1] /= math.sqrt(2)
    kinetic = 2/h**2*torch.sin(math.pi*k/(2*n))**2
    return lo+(j+.5)*h, torch.full_like(j,h), kinetic, Q


class ScreenEvolution:
    def __init__(self, kind, n, p=3):
        self.kind,self.n,self.p = kind,n,p
        self.x,self.wx,self.kx,Qx = basis(kind,n,24.,-12.,p)
        self.y,self.wy,self.ky,Qy = basis(kind,n,16.,-8.,p)
        self.Qx,self.Qy = Qx.to(torch.complex128),Qy.to(torch.complex128)
        screen = (self.x>=0)&(self.x<.75)
        holes = ((self.y>1)&(self.y<2)) | ((self.y>-2)&(self.y<-1))
        self.Ar = Qx[screen].contiguous()
        self.Br = Qy[holes].contiguous()
        psi_x = gaussian(self.x,0.,x0=-6.,k=4.,sigma=.8)*self.wx.sqrt()
        psi_y = gaussian(self.y,0.,x0=0.,k=0.,sigma=2.)*self.wy.sqrt()
        self.initial = (self.Qx.T@psi_x)[:,None]*(self.Qy.T@psi_y)[None,:]
        self.mass0 = float(self.initial.abs().square().sum())
        self.kinetic = self.kx[:,None]+self.ky[None,:]

    def kick(self, z, coefficient):
        # P_x z (I-P_holes), using orthonormal rows of Q, costs O(N^3/32).
        nx,ny = z.shape
        rank = self.Ar.shape[0]
        # Fold real/imaginary columns together so real Q rows use real GEMM.
        rows = (self.Ar@torch.view_as_real(z).reshape(nx,2*ny)).reshape(rank,ny,2)
        holes = torch.einsum('ayc,by->abc',rows,self.Br)
        rows = rows-torch.einsum('abc,by->ayc',holes,self.Br)
        projected = self.Ar.T@rows.reshape(rank,2*ny)
        return z+coefficient*torch.view_as_complex(projected.reshape(nx,ny,2))

    def evolve(self, final_time, dt):
        if dt<=0 or final_time<0:
            raise ValueError('positive dt and nonnegative final_time required')
        steps = max(1,math.ceil(final_time/dt))
        step = final_time/steps
        w = 1/(2-2**(1/3))
        coefficients = (w,1-2*w,w)
        phases = [torch.exp(-.5j*step*c*self.kinetic) for c in coefficients]
        kicks = [complex(math.cos(-30*step*c),math.sin(-30*step*c))-1 for c in coefficients]
        z = self.initial.clone()
        for _ in range(steps):
            for phase,kick in zip(phases,kicks):
                z = phase*self.kick(phase*z,kick)
        return z,dict(steps=steps,dt=step,probability_initial=self.mass0,
                      probability_final=float(z.abs().square().sum()))

    def physical(self,z):
        return (self.Qx@z@self.Qy.T)/(self.wx.sqrt()[:,None]*self.wy.sqrt()[None,:])


def interpolate_y(values, nodes, samples, kind, n, p=3):
    """Local polynomial interpolation of complex amplitude, before squaring."""
    if kind == 'fr':
        cells = ((samples+8)*n/16).floor().long().clamp(0,n-1)
        indices = cells[:,None]*(p+1)+torch.arange(p+1)[None,:]
    else:
        center = torch.searchsorted(nodes,samples)-1
        start = (center-1).clamp(0,len(nodes)-4)
        indices = start[:,None]+torch.arange(4)[None,:]
    selected = nodes[indices]
    weights = torch.ones_like(selected)
    for j in range(selected.shape[1]):
        for k in range(selected.shape[1]):
            if j != k:
                weights[:,j] *= (samples-selected[:,k])/(selected[:,j]-selected[:,k])
    return (values[:,indices]*weights[None,:,:]).sum(-1)


def observables(model, psi, samples):
    density = psi.abs().square()
    mass0 = model.mass0
    probability = float((density[model.x>=.75]*model.wx[model.x>=.75,None]*model.wy[None,:]).sum())/mass0
    mask = (model.x>=4.5)&(model.x<6.)
    strip = interpolate_y(psi[mask],model.y,samples,model.kind,model.n,model.p)
    profile = (strip.abs().square()*model.wx[mask,None]).sum(0)/mass0
    return probability,profile


def extremum(y, f, lower, upper, maximum=True):
    candidates = torch.where((y>=lower)&(y<=upper))[0]
    local = f[candidates]
    i = int(candidates[local.argmax() if maximum else local.argmin()])
    if i == int(candidates[0]) or i == int(candidates[-1]):
        raise ValueError(f'extremum hit search-window edge: {(lower,upper)}')
    # Sub-sample parabolic refinement on uniform detector coordinates.
    denominator = f[i-1]-2*f[i]+f[i+1]
    offset = float(.5*(f[i-1]-f[i+1])/denominator)
    value = float(f[i]-.25*(f[i-1]-f[i+1])*offset)
    return dict(position=float(y[i])+offset*float(y[1]-y[0]),value=value)


def fringe_metrics(y,profile):
    # Fixed physical windows avoid matching a numerical ripple as a fringe.
    central = extremum(y,profile,-.5,.5)
    peaks = dict(center=central)
    contrasts = {}
    troughs = {}
    for side,lo,hi in (('left',-3.5,-1.8),('right',1.8,3.5)):
        peak = extremum(y,profile,lo,hi)
        trough = extremum(y,profile,*sorted((central['position'],peak['position'])),maximum=False)
        peaks[side]=peak
        troughs[side]=trough
        contrasts[side]=(peak['value']-trough['value'])/(peak['value']+trough['value'])
    return dict(peaks=peaks,troughs=troughs,contrast=contrasts)


def run_one(kind,n,p,final_time,dt,output,samples):
    label=f'{kind}_n{n}_p{p}_dt{dt:g}'
    folder=output/label
    folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'solution.pt'
    config=dict(version=VERSION,kind=kind,n=n,p=p,time=final_time,requested_dt=dt)
    model=ScreenEvolution(kind,n,p)
    started=time.perf_counter()
    if cache.exists():
        saved=torch.load(cache,weights_only=True)
        if saved['config']!=config:
            raise ValueError(f'incompatible cache {cache}')
        psi,stats=saved['psi'],saved['stats']
    else:
        z,stats=model.evolve(final_time,dt)
        psi=model.physical(z)
        stats['seconds']=time.perf_counter()-started
        torch.save(dict(config=config,psi=psi,stats=stats),cache)
    probability,profile=observables(model,psi,samples)
    report=dict(label=label,**config,**stats,transmitted_probability=probability,
                fringes=fringe_metrics(samples,profile))
    report['mass_change']=report['probability_final']-report['probability_initial']
    (folder/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report,profile,model,psi


def compare(report,profile,reference,ref_profile,samples):
    result=dict(label=report['label'],transmitted_probability=report['transmitted_probability'])
    difference=report['transmitted_probability']-reference['transmitted_probability']
    result.update(transmission_absolute_error=abs(difference),
                  transmission_signed_difference=difference,
                  transmission_relative_error=abs(difference)/reference['transmitted_probability'],
                  profile_relative_l2=float(torch.linalg.vector_norm(profile-ref_profile)/torch.linalg.vector_norm(ref_profile)))
    result['peak_position_error']={key:report['fringes']['peaks'][key]['position']-reference['fringes']['peaks'][key]['position'] for key in ('left','center','right')}
    result['contrast_error']={key:report['fringes']['contrast'][key]-reference['fringes']['contrast'][key] for key in ('left','right')}
    return result


def density_samples(model,psi):
    """Render the FR polynomial inside each cell; never interpolate across faces."""
    if model.kind != 'fr':
        return model.x,model.y,psi.abs().square()
    n,k,q=model.n,model.p+1,12
    nodes=(model.x[:k]-(-12+12/n))/(12/n)
    points=(torch.arange(q,dtype=torch.float64)+.5)*2/q-1
    L=torch.ones(q,k,dtype=torch.float64)
    for j in range(k):
        for h in range(k):
            if j!=h:L[:,j]*=(points-nodes[h])/(nodes[j]-nodes[h])
    amplitude=torch.einsum('qi,rj,aibj->aqbr',L.to(psi.dtype),L.to(psi.dtype),psi.reshape(n,k,n,k)).reshape(n*q,n*q)
    indices=torch.arange(n*q,dtype=torch.float64)+.5
    return -12+24*indices/(n*q),-8+16*indices/(n*q),amplitude.abs().square()


def plots(output, runs, samples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for report,profile,_,_ in runs:
        axes[0].plot(samples,profile,label=report['label'],lw=1.2)
        axes[1].plot(samples,profile/profile.max(),label=report['label'],lw=1.2)
    for ax in axes:
        ax.set(xlabel='y',ylabel='Probability density integrated over 4.5 <= x < 6',xlim=(-5,5))
        ax.grid(alpha=.2)
    axes[1].set_ylabel('Profile normalized by its maximum')
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'fringes.png',dpi=170);plt.close(fig)
    # Standalone density cloud, sampled on its actual physical nodes.
    for report,_,model,psi in (runs[0],runs[-1]):
        fig,ax=plt.subplots(figsize=(9,5))
        plot_x,plot_y,density=density_samples(model,psi)
        image=ax.pcolormesh(plot_x.numpy(),plot_y.numpy(),density.numpy().T,shading='nearest',cmap='magma')
        for lo,hi in ((-8.,-2.),(-1.,1.),(2.,8.)):
            ax.add_patch(Rectangle((0,lo),.75,hi-lo,facecolor='cyan',alpha=.3))
        ax.axvspan(4.5,6.,facecolor='white',alpha=.12,label='Detector band')
        ax.set(xlabel='x',ylabel='y',title=f'Probability density |psi|², t={report["time"]:g}; {report["label"]}')
        fig.colorbar(image,ax=ax,label='|psi|²')
        ax.legend(loc='upper left');fig.tight_layout();fig.savefig(output/f'{report["label"]}_density.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fr',nargs='+',type=int,default=[32,64,128])
    parser.add_argument('--reference',nargs='+',type=int,default=[256,512,1024])
    parser.add_argument('--p',type=int,default=3)
    parser.add_argument('--dt',type=float,help='override both FR and reference steps')
    parser.add_argument('--fr-dt',type=float,default=.00025)
    parser.add_argument('--reference-dt',type=float,default=.0005)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--time',type=float,default=3.)
    parser.add_argument('--time-check',action='store_true')
    parser.add_argument('--output',default='results/double_slit_accuracy')
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--production-solution',help='optional saved production RK4 solution.pt for direct comparison')
    args=parser.parse_args()
    if any(n<32 or n%32 for n in args.fr+args.reference):
        parser.error('all resolutions must be multiples of 32 to align screen/slit edges')
    torch.set_num_threads(args.threads)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    samples=torch.linspace(-5.,5.,2001,dtype=torch.float64)
    runs=[]
    for kind,sizes in (('fr',args.fr),('fd',args.reference)):
        for n in sizes:
            step=args.dt if args.dt is not None else (args.fr_dt if kind=='fr' else args.reference_dt)
            runs.append(run_one(kind,n,args.p,args.time,step,output,samples))
    reference=runs[-1]
    spatial_reference=compare(runs[-2][0],runs[-2][1],reference[0],reference[1],samples) if len(args.reference)>1 else None
    time_checks=[]
    temporal_runs=[]
    if args.time_check:
        for index in (len(args.fr)-1,len(runs)-1):
            base=runs[index]
            fine=run_one(base[0]['kind'],base[0]['n'],args.p,args.time,base[0]['requested_dt']/2,output,samples)
            time_checks.append(compare(base[0],base[1],fine[0],fine[1],samples))
            temporal_runs.append(fine)
        reference=temporal_runs[-1]
    runs+=temporal_runs
    comparisons=[compare(r[0],r[1],reference[0],reference[1],samples) for r in runs]
    production=None
    if args.production_solution:
        saved=torch.load(args.production_solution,weights_only=True)
        U=saved['solution']
        n=U.shape[0];p=U.shape[2]-1
        if U.shape != (n,n,p+1,p+1,2) or abs(saved['time']-args.time)>1e-12:
            raise ValueError('production solution must be 2D and have the comparison final time')
        model=ScreenEvolution('fr',n,p)
        expected_x=model.x[:,None].expand(len(model.x),len(model.y))
        expected_y=model.y[None,:].expand_as(expected_x)
        for coords,expected in zip(saved['coords'],(expected_x,expected_y)):
            torch.testing.assert_close(coords.permute(0,2,1,3).reshape_as(expected),expected)
        psi=torch.complex(U[...,0],U[...,1]).permute(0,2,1,3).reshape(len(model.x),len(model.y))
        probability,profile=observables(model,psi,samples)
        metrics=dict(label=f'production_rk4_n{n}_p{p}',kind='production',n=n,p=p,time=args.time,transmitted_probability=probability,fringes=fringe_metrics(samples,profile))
        production=dict(metrics=metrics,comparison=compare(metrics,profile,reference[0],reference[1],samples))
        runs.insert(0,(metrics,profile,model,psi))
    report=dict(configuration=dict(time=args.time,detector_x=[4.5,6.],y_range=[-5.,5.],
                                   reference_method='cell-centred finite differences, second order, homogeneous Dirichlet',
                                   time_method='fourth-order symmetric triple-Strang composition',
                                   contrast_definition='(first side peak - intervening trough)/(first side peak + intervening trough)',
                                   probability_definition='integral(x >= .75, |psi(t)|²) / initial total probability; finite time'),
                reference=reference[0],runs=[r[0] for r in runs],comparisons=comparisons,time_checks=time_checks,
                reference_grid_difference=spatial_reference,production=production)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'profiles.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['y']+[r[0]['label'] for r in runs])
        writer.writerows(torch.stack([samples]+[r[1] for r in runs],-1).tolist())
    with (output/'comparison.csv').open('w') as f:
        columns=['label','transmitted_probability','transmission_absolute_error','transmission_relative_error','profile_relative_l2']
        writer=csv.writer(f);writer.writerow(columns+['left_peak_error','center_peak_error','right_peak_error','left_contrast_error','right_contrast_error'])
        for row in ([production['comparison']] if production else [])+comparisons:
            writer.writerow([row[k] for k in columns]+[row['peak_position_error'][k] for k in ('left','center','right')]+[row['contrast_error'][k] for k in ('left','right')])
    if not args.no_plot:plots(output,runs,samples)


if __name__=='__main__':
    main()
