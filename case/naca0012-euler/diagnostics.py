"""Streaming probes, regional residuals and conservation diagnostics."""
import csv
import json
from pathlib import Path
import torch


class Diagnostics:
    def __init__(self,airfoil,output):
        self.a=airfoil;self.s=airfoil.s;self.output=Path(output)
        s=self.s;b=s.blocks[0]
        self.center=(b['xy']*b['weight'][...,None]).sum(1)/b['areas'][:,None]
        x,y=self.center.unbind(-1)
        self.regions={
            'near':(x>-.5)&(x<1.5)&(y.abs()<.5),
            'wake':(x>=1.5)&(x<10)&(y.abs()<1),
            'outer':self.center.norm(dim=-1)>10,
        }
        coords=torch.stack(s.coords,-1)
        requested=[('upper_near',.5,.15),('lower_near',.5,-.15),('upper_1c',.5,1.),
                   ('lower_1c',.5,-1.),('upstream_5c',-5.,0.),('upstream_19c',-19.,0.),
                   ('upper_5c',.5,5.),('upper_19c',.5,19.),('lower_19c',.5,-19.),
                   ('wake_2c',2.,0.),('wake_5c',5.,0.),('wake_10c',10.,0.),
                   ('wake_20c',20.,0.),('outlet_minus_1c',float(coords[:,0].max())-1.,0.)]
        self.probes=[]
        for name,px,py in requested:
            d=(coords-torch.tensor([px,py])).norm(dim=-1);index=int(d.argmin())
            self.probes.append(dict(name=name,requested=[px,py],actual=coords[index].tolist(),
                                    distance=float(d[index]),dof_index=index))
        (self.output/'probes.json').write_text(json.dumps(self.probes,indent=2)+'\n')
        self.probe_ids=torch.tensor([p['dof_index'] for p in self.probes])
        ids=airfoil.outer_ids;n=s.normals[ids]
        self.boundary_groups={'left':n[...,0]<-.9,'right':n[...,0]>.9,
                              'top':n[...,1]>.9,'bottom':n[...,1]<-.9}
        self.initial_integral=None
        self.streams={}
        self.writers={}
        headers={
            'probes':['iteration','time','name','rho','u','v','p','p_prime','Mach','entropy_p_over_rho_gamma'],
            'diagnostics':['iteration','time','rho_min','p_min','Mach_max','wall_un_rms','wall_un_max',
                           'wall_dp_numerical_rms','mass','energy','mass_change','energy_change',
                           'mass_boundary_flux','energy_boundary_flux','mass_rhs_integral','energy_rhs_integral',
                           'mass_balance_error','energy_balance_error','residual_cell_max','worst_x','worst_y',
                           'residual_near','residual_wake','residual_outer','dt_min','dt_max'],
            'boundary':['iteration','time','side','p_prime_rms','p_prime_max_abs',
                        'acoustic_out_rms','acoustic_in_rms','mass_flux','energy_flux'],
        }
        for name,header in headers.items():
            path=self.output/(name+'.csv');exists=path.exists() and path.stat().st_size>0
            stream=path.open('a',buffering=1);writer=csv.writer(stream)
            if not exists:writer.writerow(header)
            self.streams[name]=stream;self.writers[name]=writer
        self.latest=None

    def sample(self,U,R,iteration,time,dt):
        a=self.a;s=self.s;b=s.blocks[0]
        V=U.reshape(b['ne'],b['ns'],4);Q=b['B']@V
        rho,vel,p=a.eq.primitive(Q)
        mach=vel.norm(dim=-1)/(1.4*p/rho).sqrt()
        rq=b['B']@R.reshape_as(V)
        cell_rms=((rq.square()*b['weight'][...,None]).sum(1)/b['areas'][:,None]).sqrt().amax(1)
        worst=int(cell_rms.argmax())
        regional=[]
        for mask in self.regions.values():
            regional.append(float(((rq[mask].square()*b['weight'][mask,:,None]).sum((0,1))/b['areas'][mask].sum()).sqrt().max()) if mask.any() else None)
        L=s._states(U);outside=L[s.neighbor]
        outside=torch.where(s.flip[:,None,None],outside.flip(1),outside)
        F=a.common(L,outside)
        edge_weight=s.face_jacobian*s.fw
        boundary_flux=(F[s.mesh.boundary]*edge_weight[s.mesh.boundary,:,None]).sum((0,1))
        integral=s.integral(U);rhs_integral=s.integral(R)
        if self.initial_integral is None:self.initial_integral=integral.clone()
        change=integral-self.initial_integral
        wall=L[a.wall_ids];wrho,wvel,wp=a.eq.primitive(wall)
        un=(wvel*s.normals[a.wall_ids]).sum(-1)
        ww=edge_weight[a.wall_ids]
        un_rms=float(((un**2*ww).sum()/ww.sum()).sqrt())
        dp_rms=float((((a.wall_pressure(wall,s.normals[a.wall_ids])-wp)**2*ww).sum()/ww.sum()).sqrt())
        row=[iteration,time,float(rho.min()),float(p.min()),float(mach.max()),un_rms,float(un.abs().max()),dp_rms,
             float(integral[0]),float(integral[-1]),float(change[0]),float(change[-1]),
             float(boundary_flux[0]),float(boundary_flux[-1]),float(rhs_integral[0]),float(rhs_integral[-1]),
             float((boundary_flux+rhs_integral)[0]),float((boundary_flux+rhs_integral)[-1]),
             float(cell_rms.max()),*self.center[worst].tolist(),*regional,float(dt.min()),float(dt.max())]
        self.writers['diagnostics'].writerow(row)
        pr,pv,pp=a.eq.primitive(U[self.probe_ids]);pm=pv.norm(dim=-1)/(1.4*pp/pr).sqrt()
        for i,probe in enumerate(self.probes):
            self.writers['probes'].writerow([iteration,time,probe['name'],float(pr[i]),*pv[i].tolist(),
                                            float(pp[i]),float(pp[i]-1/1.4),float(pm[i]),float(pp[i]/pr[i]**1.4)])
        ids=a.outer_ids;nr=s.normals[ids]
        br,bv,bp=a.eq.primitive(L[ids]);bc=(1.4*bp/br).sqrt()
        dp=bp-1/1.4;du=((bv-a.far[1:3])*nr).sum(-1)
        outgoing=.5*(dp+br*bc*du);incoming=.5*(dp-br*bc*du)
        for name,mask in self.boundary_groups.items():
            weights=edge_weight[ids][mask];area=weights.sum()
            def rms(X):return float((X[mask].square().mul(weights).sum()/area).sqrt())
            flux=(F[ids][mask]*weights[:,None]).sum(0)
            self.writers['boundary'].writerow([iteration,time,name,rms(dp),float(dp[mask].abs().max()),
                                               rms(outgoing),rms(incoming),float(flux[0]),float(flux[-1])])
        self.latest=dict(iteration=iteration,time=time,residual_near=regional[0],residual_outer=regional[2],
                         mass_balance_error=row[16],energy_balance_error=row[17],wall_un_rms=un_rms)
        for stream in self.streams.values():stream.flush()
        return self.latest

    def snapshot(self,U,R,iteration,time):
        a=self.a;s=self.s;b=s.blocks[0]
        Q=b['B']@U.reshape(b['ne'],b['ns'],4)
        rho,vel,p=a.eq.primitive(Q)
        w=b['weight'];area=b['areas']
        def mean(X):return (X*w).sum(1)/area
        residual=((b['B']@R.reshape(b['ne'],b['ns'],4)).square()*w[...,None]).sum(1)/area[:,None]
        path=self.output/'snapshots';path.mkdir(exist_ok=True)
        data=torch.stack((self.center[:,0],self.center[:,1],mean(rho),mean(vel[...,0]),mean(vel[...,1]),mean(p),
                          mean(vel.norm(dim=-1)/(1.4*p/rho).sqrt()),mean(p/rho**1.4),residual.sqrt().amax(1)),1)
        import numpy as np
        np.savetxt(path/f'cells_{iteration:07d}.csv',data.numpy(),delimiter=',',comments='',header='x,y,rho,u,v,p,Mach,entropy,residual_rms_max')
        torch.save(dict(U=U,time=time,iteration=iteration),path/f'state_{iteration:07d}.pt')

    def close(self):
        for stream in self.streams.values():stream.close()
