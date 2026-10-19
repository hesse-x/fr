"""Bounded zero-angle diagnostics; does not start or resume the polar sweep."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from airfoil import Airfoil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).parent/'results'
    p.add_argument('--checkpoint',type=Path,default=base/'reflect0/solution.pt')
    p.add_argument('--output',type=Path,default=base/'diagnose_zero')
    p.add_argument('--compare-steps',type=int,default=200)
    a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/fr-naca-inductor')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-naca-mpl')
    a.output.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,weights_only=False);U=ck['U'];m=ck['stats']
    f=Airfoil(m['mesh'],p=m['p'],alpha=0,mach=m['mach'],quadrature_order=m['quadrature'])
    s=f.s;b=s.blocks[0]
    report={'checkpoint':str(a.checkpoint),'saved_steps':m['steps'],'saved_pseudo_time_min':m['pseudo_time_min']}
    with torch.no_grad():
        R=f.rhs(U)
        V=U.reshape(b['ne'],b['ns'],4)
        q=b['B']@V
        cellr=((b['B']@R.reshape(b['ne'],b['ns'],4)).square()*b['weight'][...,None]).sum(1)/b['areas'][:,None]
        centers=(b['xy']*b['weight'][...,None]).sum(1)/b['areas'][:,None]
        near=((centers[:,0]>.0)&(centers[:,0]<1)&(centers[:,1].abs()<.25))
        dt=f.local_dt(U,3.).flatten()
        report.update(residual_rms=float(s.norm(R).max()),rho_min=float(q[...,0].min()),
                      pressure_min=float(f.eq.primitive(q)[2].min()),valid=f.valid(U),
                      local_dt_min=float(dt.min()),local_dt_max=float(dt.max()),
                      near_body_dt_median=float(dt[near].median()),
                      dt_ratio_max_min=float(dt.max()/dt.min()),
                      residual_near_body=float(cellr[near].mean(0).sqrt().max()),
                      residual_far=float(cellr[centers.norm(dim=-1)>10].mean(0).sqrt().max()))
        history=np.array(ck['history'])
        report['residual_initial']=float(history[0,4]);report['residual_reduction_factor']=float(history[0,4]/history[-1,4])
        # Geometric free-stream identity: allow uniform flow through every face.
        old_wall=f.wall.clone();old_outer=f.outer_ids.clone()
        f.wall[:]=False;f.outer_ids=torch.empty(0,dtype=torch.long)
        uniform=f.far.expand_as(U).clone()
        report['free_stream_max_abs_rhs']=float(f.rhs(uniform).abs().max())
        f.wall=old_wall;f.outer_ids=old_outer
        report['optimized_vs_repository_max_abs']=float((f.rhs(U)-s.rhs(U)).abs().max())
        finer=Airfoil(m['mesh'],p=m['p'],alpha=0,mach=m['mach'],quadrature_order=8)
        r8=finer.rhs(U)
        report['quadrature_6_vs_8_relative_residual']=float(s.norm(R-r8).max()/s.norm(R).max())
        report['quadrature_6_vs_8_loads']={'q6':f.loads(U)[:2],'q8':finer.loads(U)[:2]}
        normals=s.normals[f.wall_ids];weights=s.face_jacobian[f.wall_ids]*s.fw
        report['wall_integrated_normal']=((normals*weights[...,None]).sum((0,1))).tolist()
        # Surface point set should have mirror symmetry; triangles need not.
        xy=s.face_coordinates[f.wall_ids].reshape(-1,2)
        mirror=xy*torch.tensor([1.,-1.])
        report['wall_mirror_point_distance_max']=float(torch.cdist(xy,mirror).min(1).values.max())
        vertices=torch.cat([block.geometry[:,:3,:].reshape(-1,2) for block in f.mesh.blocks]).unique(dim=0)
        rounded={tuple(row) for row in np.round(vertices.numpy(),9)}
        mirrored=np.round((vertices*torch.tensor([1.,-1.])).numpy(),9)
        report['mesh_mirror_vertex_fraction']=sum(tuple(row) in rounded for row in mirrored)/len(vertices)
        print(json.dumps(report,indent=2),flush=True)
        (a.output/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
        with (a.output/'cells.csv').open('w') as stream:
            w=csv.writer(stream);w.writerow(['x','y','residual_rms_max','local_dt'])
            w.writerows(torch.cat((centers,cellr.sqrt().amax(1)[:,None],dt[:,None]),1).tolist())
        comparisons={}
        for cfl,steps in [(3.,a.compare_steps),(1.,3*a.compare_steps)]:
            state=U.clone();pseudo=0.;rows=[];start=time.monotonic()
            rhs=torch.compile(f.rhs,fullgraph=True)
            for i in range(steps):
                h=f.local_dt(state,cfl);pseudo+=float(h.min())
                state=f.rk4(state,h,rhs)
                if (i+1)%50==0 or i+1==steps:
                    cl,cd,_=f.loads(state)
                    rows.append([i+1,pseudo,cl,cd,float(s.norm(rhs(state)).max())])
                    print(f'controlled CFL={cfl:g} step={i+1}/{steps} R={rows[-1][-1]:.5e}',flush=True)
            comparisons[str(cfl)]={'steps':steps,'pseudo_time_min_increment':pseudo,'valid':f.valid(state),
                                   'CL':f.loads(state)[0],'CD':f.loads(state)[1],
                                   'residual':float(s.norm(rhs(state)).max()),'seconds':time.monotonic()-start}
            with (a.output/f'cfl_{cfl:g}.csv').open('w') as stream:
                w=csv.writer(stream);w.writerow(['iteration','pseudo_time_min_increment','CL','CD','residual']);w.writerows(rows)
            if cfl==3.:coarse=state.clone()
            else:report['cfl_comparison_state_rms_difference']=s.norm(state-coarse).tolist()
        report['controlled_runs']=comparisons
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(history[:,0],history[:,2],label='CL');axes[0,0].plot(history[:,0],history[:,3],label='CD pressure')
    axes[0,0].set_xlabel('Explicit RK4 iterations');axes[0,0].legend();axes[0,0].grid(alpha=.3)
    axes[0,1].semilogy(history[:,0],history[:,4]);axes[0,1].set_xlabel('Explicit RK4 iterations');axes[0,1].set_ylabel('Residual RMS max');axes[0,1].grid(alpha=.3)
    cl,cd,pressure=f.loads(U);cp=(pressure-1/1.4)/(.5*f.mach**2)
    for mask,label in [(xy[:,1]>=0,'upper'),(xy[:,1]<0,'lower')]:
        x=xy[:,0][mask].numpy();v=cp.flatten()[mask].numpy();ids=np.argsort(x)
        axes[1,0].plot(x[ids],v[ids],label=label)
    axes[1,0].invert_yaxis();axes[1,0].set_xlabel('x/c');axes[1,0].set_ylabel('Cp (not steady)');axes[1,0].legend();axes[1,0].grid(alpha=.3)
    import matplotlib.colors as colors
    sc=axes[1,1].scatter(centers[:,0],centers[:,1],c=cellr.sqrt().amax(1),s=5,norm=colors.LogNorm(),cmap='magma')
    axes[1,1].set_xlim(-1,2);axes[1,1].set_ylim(-1,1);axes[1,1].set_aspect('equal');axes[1,1].set_title('Cell residual near airfoil');fig.colorbar(sc,ax=axes[1,1])
    fig.tight_layout();fig.savefig(a.output/'diagnostics.png',dpi=180);plt.close(fig)
    (a.output/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
