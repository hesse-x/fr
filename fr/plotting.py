"""Cell-local sampling and PNG/GIF output; plotting dependencies are loaded lazily."""
import math
import os
from pathlib import Path
import tempfile
import torch


def add_plot_arguments(parser, *, unsteady=True):
    parser.add_argument('--no-plot', action='store_true', help='disable PNG/GIF output')
    if unsteady:
        parser.add_argument('--frames', type=int, default=60, help='maximum GIF frames, including endpoints; 0 disables GIF')
        parser.add_argument('--fps', type=int, default=12, help='GIF playback frames per second')


def _lagrange(nodes, points):
    result=torch.ones(len(points),len(nodes),dtype=nodes.dtype,device=nodes.device)
    for j in range(len(nodes)):
        for k in range(len(nodes)):
            if j!=k: result[:,j]*=(points-nodes[k])/(nodes[j]-nodes[k])
    return result


class FieldSampler:
    """Sample each DG cell independently, preserving jumps and mesh boundaries.

    3D Cartesian data are interpolated onto the physical middle-z plane.
    Curved cells use subdivided reference triangles mapped through their geometry.
    """
    def __init__(self, solver):
        self.solver=solver
        self.dim=solver.dim
        self.note=''
        n=max(4,2*solver.p+2)
        self.parts=[]
        if hasattr(solver,'blocks'):
            polygons=[]
            for b in solver.blocks:
                tri=[]
                for i in range(n):
                    for j in range(n if b['basis'].kind=='quad' else n-i):
                        a=(i/n,j/n); c=((i+1)/n,j/n); d=(i/n,(j+1)/n)
                        tri.append((a,c,d))
                        if b['basis'].kind=='quad' or i+j<n-1:
                            tri.append((c,((i+1)/n,(j+1)/n),d))
                ref=torch.tensor(tri,dtype=torch.float64,device=solver.device)
                xy=torch.einsum('qi,eik->eqk',b['geometry'].evaluate(ref.reshape(-1,2)),b['X'])
                polygons.append(xy.reshape(-1,3,2).cpu())
                self.parts.append((b,b['basis'].evaluate(ref.mean(1))))
            self.polygons=torch.cat(polygons).numpy()
        else:
            device=solver.z.device
            mid=(torch.arange(n,dtype=torch.float64,device=device)+.5)*2/n-1
            self.L=_lagrange(solver.z,mid)
            self.n=n
            self.zslice=None
            if self.dim==3:
                lo,hi=solver.bounds[2]; z=(lo+hi)/2
                cell=min(solver.cells[2]-1,int((z-lo)/solver.h[2]))
                ref=2*(z-lo-cell*solver.h[2])/solver.h[2]-1
                self.zslice=(cell,_lagrange(solver.z,torch.tensor([ref],dtype=torch.float64,device=device))[0])
                self.note=f' (xy slice, z={z:g})'
            edges=[torch.linspace(lo,hi,count*n+1,dtype=torch.float64) for count,(lo,hi) in zip(solver.cells,solver.bounds)]
            if self.dim==1:
                self.x=((edges[0][:-1]+edges[0][1:])/2).numpy()
            else:
                x,y=torch.meshgrid(edges[0],edges[1],indexing='ij')
                grid=torch.stack((x,y),-1)
                self.polygons=torch.stack((grid[:-1,:-1],grid[1:,:-1],grid[1:,1:],grid[:-1,1:]),-2).reshape(-1,4,2).numpy()

    def state(self,U):
        if self.parts:
            return torch.cat([(L.to(U.dtype)@U[b['offset']].reshape(b['ne'],b['ns'],-1)).reshape(-1,U.shape[-1]) for b,L in self.parts])
        if self.dim==3:
            cell,Lz=self.zslice
            U=torch.einsum('abijkv,k->abijv',U[:,:,cell],Lz)
        if self.dim==1:
            return torch.einsum('qi,eiv->eqv',self.L,U).reshape(-1,U.shape[-1])
        return torch.einsum('qi,rj,abijv->aqbrv',self.L,self.L,U).reshape(-1,U.shape[-1])


class PlotRecorder:
    """Accepted-state observer; stores only sampled plot fields, never RK stages.

    At most `frames` snapshots, approximately uniformly spaced in physical time.
    Endpoints are always retained. Observation never changes the solver time step.
    """
    def __init__(self,solver,fields,final_time,*,frames=60,fps=12,enabled=True):
        if frames<0 or frames==1 or fps<=0: raise ValueError('frames must be 0 or >=2; fps must be positive')
        self.enabled=enabled
        self.fields=fields
        self.final_time=final_time
        self.frames,self.fps=frames,fps
        self.times=[]; self.values=[]
        self.next_index=0
        self.sampler=FieldSampler(solver) if enabled else None

    def __call__(self,U,t):
        if not self.enabled or self.frames==0: return
        tol=1e-12*max(1.,abs(self.final_time))
        if self.times and abs(t-self.times[-1])<=tol: return
        if self.next_index>=self.frames: return
        target=self.final_time*self.next_index/(self.frames-1)
        if t+tol<target: return
        # Skip crossed targets instead of duplicating a state after a large step.
        self.next_index=min(self.frames,math.floor((t+tol)/self.final_time*(self.frames-1))+1) if self.final_time>0 else self.frames
        self.times.append(float(t)); self.values.append(self.sample(U))

    def sample(self,U):
        with torch.no_grad():
            V=self.sampler.state(U)
            return [func(V).detach().cpu().numpy().copy() for func in self.fields.values()]

    def finish(self,stem,U,t):
        if not self.enabled: return
        stem=Path(stem); stem.parent.mkdir(parents=True,exist_ok=True)
        self(U,t)
        final=self.sample(U)
        os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'fr-matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.animation import FuncAnimation, PillowWriter
        frames=self.values or [final]
        limits=[]
        for k in range(len(self.fields)):
            low=min(float(v[k].min()) for v in frames+[final]); high=max(float(v[k].max()) for v in frames+[final])
            if low<0<high: low,high=-max(abs(low),abs(high)),max(abs(low),abs(high))
            if high-low<1e-12*max(1.,abs(high)): low-=1e-6*max(1.,abs(high)); high+=1e-6*max(1.,abs(high))
            limits.append((low,high))
        fig,axes=plt.subplots(1,len(self.fields),figsize=(5*len(self.fields),4.4),squeeze=False,layout='constrained')
        artists=[]
        for k,(ax,label,vals,(lo,hi)) in enumerate(zip(axes[0],self.fields,final,limits)):
            if self.sampler.dim==1:
                artist,=ax.plot(self.sampler.x,vals,lw=1.5)
                ax.set_ylim(lo,hi); ax.set_ylabel(label)
            else:
                artist=PolyCollection(self.sampler.polygons,array=vals,cmap='RdBu_r' if lo<0<hi else 'viridis',clim=(lo,hi),edgecolors='none',antialiased=False,rasterized=True)
                ax.add_collection(artist); ax.autoscale_view(); ax.set_aspect('equal'); ax.set_ylabel('y')
                fig.colorbar(artist,ax=ax,shrink=.85,label=label)
            ax.set_xlabel('x'); ax.set_title(label+self.sampler.note); artists.append(artist)
        title=fig.suptitle(f't = {t:.6g}')
        fig.savefig(stem.with_suffix('.png'),dpi=150)
        if self.frames and len(self.times)>1:
            def update(i):
                title.set_text(f't = {self.times[i]:.6g}')
                for artist,vals in zip(artists,self.values[i]):
                    if self.sampler.dim==1: artist.set_ydata(vals)
                    else: artist.set_array(vals)
                return artists+[title]
            animation=FuncAnimation(fig,update,frames=len(self.times),interval=1000/self.fps,blit=False)
            animation.save(stem.with_suffix('.gif'),writer=PillowWriter(fps=self.fps),dpi=90)
        plt.close(fig)
        import json
        stem.with_name(stem.name+'_plot.json').write_text(json.dumps(dict(snapshot_times=self.times,color_limits=dict(zip(self.fields,limits)),slice=self.sampler.note,fps=self.fps),indent=2)+'\n')


def euler_fields(equation):
    return {'Density':lambda U:U[...,0], 'Pressure':lambda U:equation.primitive(U)[2],
            'Mach':lambda U:equation.primitive(U)[1].norm(dim=-1)/(equation.gamma*equation.primitive(U)[2]/U[...,0]).sqrt()}
