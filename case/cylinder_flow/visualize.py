"""Accepted-state vorticity recording and curved-cell wake PNG/GIF output."""
import json
import math
import os
from pathlib import Path

import torch
from fr.plotting import FieldSampler


class WakeRecorder:
    def __init__(self,physics,args,*,frames=120,fps=15,start=0.,limit=3.):
        if args.time_mode!='global':
            raise ValueError('vorticity animation requires global physical time stepping')
        if frames==1 or frames<0 or fps<=0 or not math.isfinite(start) or start<0 or not math.isfinite(limit) or limit<=0:
            raise ValueError('frames must be 0 or >=2, fps/limit positive, and start nonnegative')
        self.a=physics; self.output=Path(args.output)
        self.frames=frames; self.fps=fps; self.start=start; self.stop=args.time
        self.limit=limit; self.times=[]; self.values=[]; self.next_index=0
        self.sampler=FieldSampler(physics.s)
        xy=torch.from_numpy(self.sampler.polygons)
        lo=xy.amin(1); hi=xy.amax(1)
        self.mask=((hi[:,0]>=-2)&(lo[:,0]<=14)&(hi[:,1]>=-4)&(lo[:,1]<=4)).numpy()
        self.polygons=self.sampler.polygons[self.mask]

    def sample(self,U):
        grad=self.a.gradient(self.a.primitive_vector(U))
        omega=(grad[:,1,0]-grad[:,0,1])/self.a.mach
        return self.sampler.state(omega[:,None])[:,0].cpu().numpy()[self.mask].copy()

    def __call__(self,U,t):
        conv=t*self.a.mach
        if self.frames==0 or conv<self.start or len(self.times)>=self.frames:
            return
        target=self.start+(self.stop-self.start)*self.next_index/max(self.frames-1,1)
        if conv+1e-12<target or (self.times and conv<=self.times[-1]+1e-12):
            return
        self.times.append(conv); self.values.append(self.sample(U))
        if self.stop>self.start:
            self.next_index=max(self.next_index+1,math.floor((conv-self.start)/(self.stop-self.start)*(self.frames-1))+1)
        else:
            self.next_index=self.frames

    def finish(self,U,t):
        os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation,PillowWriter
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Circle
        self.output.mkdir(parents=True,exist_ok=True)
        conv=t*self.a.mach
        final=self.sample(U)
        if self.frames and conv>=self.start and (not self.times or conv>self.times[-1]+1e-12):
            if len(self.times)>=self.frames:
                self.times[-1]=conv; self.values[-1]=final
            else:
                self.times.append(conv); self.values.append(final)
        fig,ax=plt.subplots(figsize=(11,4.4),layout='constrained')
        artist=PolyCollection(self.polygons,array=final,cmap='RdBu_r',
            clim=(-self.limit,self.limit),edgecolors='none',antialiased=False)
        ax.add_collection(artist); ax.add_patch(Circle((0,0),.5,color='#273849',zorder=3))
        ax.set(xlim=(-2,14),ylim=(-4,4),xlabel='x / D',ylabel='y / D',aspect='equal')
        fig.colorbar(artist,ax=ax,shrink=.85,label=r'Vorticity $\omega_z D/U_\infty$ (clipped)')
        def title(time):
            return f'Cylinder wake | Re={self.a.reynolds:g}, Ma={self.a.mach:g} | t U/D = {time:.2f}'
        heading=ax.set_title(title(conv))
        fig.savefig(self.output/'vorticity.png',dpi=150)
        if len(self.times)>1:
            def update(i):
                artist.set_array(self.values[i]); heading.set_text(title(self.times[i]))
                return artist,heading
            movie=FuncAnimation(fig,update,frames=len(self.times),interval=1000/self.fps)
            movie.save(self.output/'vorticity.gif',writer=PillowWriter(fps=self.fps),dpi=100)
        plt.close(fig)
        (self.output/'vorticity_plot.json').write_text(json.dumps(dict(
            convective_times=self.times,fps=self.fps,color_limits=[-self.limit,self.limit],
            normalization='omega_z D/U_inf',view=[[-2,14],[-4,4]],
            frames=len(self.times),requested_frames=self.frames,
            reynolds=self.a.reynolds,mach=self.a.mach,p=self.a.s.p,
            note='accepted global-time states; curved-cell-local interpolation; fixed clipped color scale'),indent=2)+'\n')
