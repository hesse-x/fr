"""FR face sampling on stationary closed FW-H contours; ASCII interchange."""
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch


class FWHRecorder:
    def __init__(self,a,args,filename,start=0.,interval=.05,envelopes=None):
        if args.time_mode!='global' or not math.isfinite(interval) or not math.isfinite(start) or interval<=0 or start<0:
            raise ValueError('FW-H requires global physical time, nonnegative start and positive sample interval')
        self.a=a;self.start=start;self.interval=interval;self.index=0
        self.origin=None;self.previous=None;self.previous_time=None
        self.path=Path(filename);self.path.parent.mkdir(parents=True,exist_ok=True)
        s=a.s;b=s.blocks[0]
        owner=torch.empty(a.mesh.nfaces,dtype=torch.long)
        owner[b['ids']]=torch.arange(b['ne'])[:,None].expand_as(b['ids'])
        center=b['X'][:,:3 if b['basis'].kind=='tri' else 4].mean(1)
        sets=[('wall',a.wall_ids)]
        envelopes=envelopes if envelopes is not None else [('inner',-1.5,4.,1.),('outer',-2.,6.,1.5)]
        for name,xmin,xmax,ymax in envelopes:
            selected=(center[:,0]>=xmin)&(center[:,0]<=xmax)&(center[:,1].abs()<=ymax)
            faces=selected[owner]&~selected[owner[s.neighbor]]&~a.mesh.boundary
            ids=torch.where(faces)[0]
            if not len(ids):raise ValueError('empty FW-H contour')
            # Closed flux surface: a constant vector must integrate to zero.
            closure=(s.normals[ids]*s.face_jacobian[ids,:,None]*s.fw[None,:,None]).sum((0,1))
            if float(closure.abs().max())>1e-9:raise ValueError(f'nonclosed FW-H contour: {name}')
            sets.append((name,ids))
        self.ids=torch.cat([ids for _,ids in sets])
        self.wall_count=len(sets[0][1])*len(s.fw)
        normals=s.normals[self.ids].clone();normals[:len(a.wall_ids)]*=-1
        points=s.face_coordinates[self.ids]
        weights=s.face_jacobian[self.ids]*s.fw
        self.names=[name for name,_ in sets]
        self.labels=np.concatenate([np.full(len(ids)*len(s.fw),i) for i,(_,ids) in enumerate(sets)])
        self.points=points.reshape(-1,2).numpy();self.normals=normals.reshape(-1,2).numpy()
        self.weights=weights.flatten().numpy()
        self.file=self.path.open('w',buffering=1024*1024)
        metadata=dict(format='FR-FWH2D-1',surface_names=self.names,
            rho0=1.,c0=1.,p0=1/1.4,mean_velocity=a.far[1:3].tolist(),
            mach=a.mach,reynolds=a.reynolds,p=s.p,quadrature=len(s.fw),
            interval_convective=interval,mesh=str(args.mesh),
            mesh_sha256=hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest(),
            restart=str(args.restart) if args.restart else None,
            units='D=rho_inf=c_inf=1; t is physical solver time',
            source_flux='FR numerical conservative normal mass/momentum flux; wall normal points into fluid',
            contours='wall and closed mesh-aligned cell-union envelopes; normals outward from enclosed volume')
        self.file.write('# '+json.dumps(metadata)+'\n')
        for i,(sid,xy,n,w) in enumerate(zip(self.labels,self.points,self.normals,self.weights)):
            self.file.write(f'# point {i} {sid} {xy[0]:.17g} {xy[1]:.17g} {n[0]:.17g} {n[1]:.17g} {w:.17g}\n')
        self.file.write('# columns: time point rho u v p mass_flux momentum_flux_x momentum_flux_y\n')

    def sample(self,U,t):
        a=self.a;s=a.s
        L=s._states(U);R=s.neighbor_states(L)
        # Use the same conservative face fluxes as the FR residual: this avoids
        # introducing a separate mass-leaking interpolation on a fictitious surface.
        conv=a.common(L,R)
        visc=a.viscous(U)[1]
        flux=conv[..., :3].clone();flux[...,1:3]-=visc[...,1:3]
        flux=flux[self.ids].reshape(-1,3)
        flux[:self.wall_count]*=-1
        trace=.5*(L[self.ids]+R[self.ids]).reshape(-1,4)
        rho,velocity,pressure=a.eq.primitive(trace)
        pressure[:self.wall_count]=a.wall_pressure(L[a.wall_ids],s.normals[a.wall_ids]).flatten()
        velocity[:self.wall_count]=0.
        rows=torch.cat((torch.full((len(trace),1),t,dtype=U.dtype),
            torch.arange(len(trace),dtype=U.dtype)[:,None],rho[:,None],velocity,
            pressure[:,None],flux),-1).numpy()
        np.savetxt(self.file,rows,fmt=['%.12e','%d']+['%.12e']*7)
        self.file.flush()

    def __call__(self,U,t):
        conv=t*self.a.mach
        if self.origin is None and conv>=self.start:
            self.origin=max(self.start,conv if self.previous is None else self.start)
        while self.origin is not None:
            target=self.origin+self.index*self.interval
            if conv+1e-12<target:break
            physical=target/self.a.mach
            if self.previous is not None and t>self.previous_time:
                fraction=(physical-self.previous_time)/(t-self.previous_time)
                state=self.previous+fraction*(U-self.previous)
            else:state=U
            self.sample(state,physical);self.index+=1
        self.previous=U;self.previous_time=t

    def finish(self,U,t):
        self.file.close()
        print(f'FW-H: {self.index} uniformly spaced samples written to {self.path}',flush=True)
