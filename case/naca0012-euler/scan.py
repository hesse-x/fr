"""Run 0--15 degree explicit RK cases and plot only verified steady values."""
import argparse
import concurrent.futures
import csv
import json
import os
import queue
from pathlib import Path
import subprocess
import sys
import threading
import numpy as np

BASE=Path(__file__).resolve().parent


def plot_results(output,alphas):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-naca-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]; states=[]
    for alpha in alphas:
        folder=output/f'alpha_{alpha:02d}'
        if not (folder/'metrics.json').exists():
            rows.append(dict(alpha_deg=alpha,converged=False,status='missing',CL=None,CD_pressure=None,Cf=None))
            continue
        m=json.loads((folder/'metrics.json').read_text()); ok=m['steady_convergence_verified']
        rows.append(dict(alpha_deg=alpha,converged=ok,status=m['status'],
                         CL=m['CL'] if ok else None,CD_pressure=m['CD_pressure'] if ok else None,
                         Cf=0. if ok else None))
        states.append((alpha,folder,m))
    with (output/'polar.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['alpha_deg','converged','status','CL','CD_pressure','Cf']);w.writeheader();w.writerows(rows)
    (output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    x=[r['alpha_deg'] for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['CL','CD_pressure','Cf'],['Lift coefficient','Pressure drag coefficient','Skin friction (Euler: zero)']):
        # NaNs break lines across unverified angles.
        y=[r[key] if r[key] is not None else np.nan for r in rows]
        ax.plot(x,y,'o-',ms=4);ax.set_xlabel('Angle of attack (deg)');ax.set_ylabel(key)
        ax.set_title(title);ax.grid(alpha=.3)
    fig.suptitle('NACA 0012, M=0.3: verified steady values only')
    fig.tight_layout();fig.savefig(output/'polar.png',dpi=180);fig.savefig(output/'polar.svg');plt.close(fig)
    good=[state for state in states if state[2]['steady_convergence_verified']]
    fig,axes=plt.subplots(4,4,figsize=(15,11),sharex=True)
    for ax,alpha in zip(axes.flat,alphas):
        state=next((state for state in good if state[0]==alpha),None)
        if state:
            cp=np.genfromtxt(state[1]/'cp.csv',delimiter=',',names=True)
            for mask,label in [(cp['y']>=0,'upper'),(cp['y']<0,'lower')]:
                xx=cp['x'][mask]; yy=cp['Cp'][mask];ids=np.argsort(xx)
                ax.plot(xx[ids],yy[ids],lw=1,label=label)
            ax.legend(fontsize=7)
        else: ax.text(.5,.5,'Not converged',transform=ax.transAxes,ha='center')
        ax.set_title(f'{alpha} deg');ax.invert_yaxis();ax.grid(alpha=.2)
        ax.set_xlabel('x/c');ax.set_ylabel('Cp')
    fig.tight_layout();fig.savefig(output/'cp_by_alpha.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    cmap=plt.get_cmap('viridis')
    for alpha,folder,_ in good:
        cp=np.genfromtxt(folder/'cp.csv',delimiter=',',names=True)
        for ax,mask in zip(axes,[cp['y']>=0,cp['y']<0]):
            ids=np.argsort(cp['x'][mask]);ax.plot(cp['x'][mask][ids],cp['Cp'][mask][ids],color=cmap(alpha/15),lw=1)
    for ax,title in zip(axes,['Upper surface','Lower surface']):
        ax.invert_yaxis();ax.set_xlabel('x/c');ax.set_ylabel('Cp');ax.set_title(title);ax.grid(alpha=.2)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,15),cmap=cmap),ax=axes,label='Angle of attack (deg)')
    fig.savefig(output/'cp_overlay.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # A Cp(alpha) curve needs a specified chord location: interpolate both
    # surfaces at fixed x/c, rather than silently collapsing Cp to one number.
    locations=[.01,.05,.1,.25,.5,.75,.95]
    cp_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,side in zip(axes,['upper','lower']):
        for station in locations:
            ys=[]
            for alpha in alphas:
                state=next((state for state in good if state[0]==alpha),None)
                value=np.nan
                if state:
                    cp=np.genfromtxt(state[1]/'cp.csv',delimiter=',',names=True)
                    mask=cp['y']>=0 if side=='upper' else cp['y']<0
                    ids=np.argsort(cp['x'][mask]);xx=cp['x'][mask][ids];yy=cp['Cp'][mask][ids]
                    if xx[0]<=station<=xx[-1]: value=float(np.interp(station,xx,yy))
                ys.append(value)
                cp_rows.append([alpha,side,station,None if np.isnan(value) else value])
            ax.plot(alphas,ys,'o-',ms=2,label=f'x/c={station:g}')
        ax.invert_yaxis();ax.set_xlabel('Angle of attack (deg)');ax.set_ylabel('Cp')
        ax.set_title(side+' surface');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'cp_vs_alpha.png',dpi=180);plt.close(fig)
    with (output/'cp_vs_alpha.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['alpha_deg','surface','x_over_c','Cp']);w.writerows(cp_rows)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for alpha,folder,m in states:
        h=np.genfromtxt(folder/'history.csv',delimiter=',',names=True)
        h=np.atleast_1d(h)
        label=f'{alpha} deg'+('' if m['steady_convergence_verified'] else ' (pending)')
        axes[0].plot(h['iteration'],h['CL'],lw=.8,label=label)
        axes[1].plot(h['iteration'],h['CD_pressure'],lw=.8)
        axes[2].semilogy(h['iteration'],h['residual_rms_max'],lw=.8)
    for ax,title in zip(axes,['CL','CD pressure','Residual RMS max']):
        ax.set_xlabel('Explicit RK iterations');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(output/'convergence.png',dpi=180);plt.close(fig)
    verified=sum(r['converged'] for r in rows)
    report=['# NACA 0012 显式 RK 攻角扫描', '',f'已通过收敛条件：{verified}/{len(rows)} 个攻角。',
            '', '空间 p=3，四阶段显式 RK；Mach=0.3；具体方法和时间模式见各算例 metrics.json。',
            'Cf 为 Euler 模型的零摩擦系数；CD 仅为压力阻力。',
            '未收敛攻角在 polar.csv 中留空，不作为最终值。',
            '此处收敛表示固定网格的迭代收敛，未证明网格无关性或失速预测能力。',
            '', '| 攻角 | 收敛 | CL | CD_pressure |', '|---:|:---:|---:|---:|']
    for row in rows:
        values=[f'{row[k]:.8f}' if row[k] is not None else '—' for k in ['CL','CD_pressure']]
        report.append(f'| {row["alpha_deg"]} | {"是" if row["converged"] else "否"} | {values[0]} | {values[1]} |')
    (output/'report.md').write_text('\n'.join(report)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,default=BASE/'results/mesh/moderate.msh')
    p.add_argument('--output',type=Path,default=BASE/'results/polar')
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=15)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--threads',type=int,default=8,help='CPU threads within each case')
    p.add_argument('--max-steps',type=int,default=200000)
    p.add_argument('--cfl',type=float,default=3.)
    p.add_argument('--residual-tol',type=float,default=1e-5)
    p.add_argument('--residual-relative-tol',type=float,default=1.)
    p.add_argument('--initial-restart',type=Path)
    p.add_argument('--plot-only',action='store_true')
    a=p.parse_args()
    if not 0<=a.start<=a.end<=15 or a.workers<1 or a.threads<1: p.error('0 <= start <= end <= 15, workers >= 1 required')
    a.output.mkdir(parents=True,exist_ok=True)
    alphas=list(range(a.start,a.end+1))
    if not a.plot_only:
        settings=dict(mesh=str(a.mesh.resolve()),cfl=a.cfl,rk='ssprk43',p=3,quadrature=6,time_mode='global',
                      residual_absolute=a.residual_tol,residual_relative=a.residual_relative_tol,
                      start=a.start,end=a.end,workers=a.workers,threads=a.threads,max_steps=a.max_steps,
                      initial_restart=str(a.initial_restart.resolve()) if a.initial_restart else None)
        (a.output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        plot_lock=threading.Lock()
        def worker(chunk):
            previous=a.initial_restart
            for alpha in chunk:
                folder=a.output/f'alpha_{alpha:02d}'
                checkpoint=folder/'solution.pt'
                metric=folder/'metrics.json'
                if metric.exists():
                    m=json.loads(metric.read_text())
                    r=m['convergence']['residual']
                    if (m['steady_convergence_verified'] and m['convergence'].get('criterion')=='residual'
                        and r<=a.residual_tol and r<=m['initial_residual_reference']*a.residual_relative_tol
                        and m['method']=='explicit_ssprk43' and m['p']==3 and m['quadrature']==6
                        and m['mesh']==str(a.mesh.resolve())):
                        previous=checkpoint;continue
                command=[sys.executable,str(BASE/'solve.py'),'--mesh',str(a.mesh),'--alpha',str(alpha),
                         '--output',str(folder),'--max-steps',str(a.max_steps),'--cfl',str(a.cfl),
                         '--threads',str(a.threads),'--rk','ssprk43','--p','3','--quadrature','6','--time-mode','global',
                         '--stop-criterion','residual','--residual-tol',str(a.residual_tol),
                         '--residual-relative-tol',str(a.residual_relative_tol)]
                restart=checkpoint if checkpoint.exists() else previous
                if restart: command+=['--restart',str(restart)]
                with (a.output/f'alpha_{alpha:02d}.log').open('a') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f'alpha {alpha} failed; see {log.name}')
                m=json.loads(metric.read_text())
                if m['steady_convergence_verified']:previous=checkpoint
                with plot_lock:plot_results(a.output,alphas)
                print(f'completed alpha {alpha}: '+json.loads(metric.read_text())['status'],flush=True)
        # Shared work queue keeps both slots busy when angles take different
        # times, rather than reserving a fixed range for each worker.
        pending=queue.Queue()
        for alpha in alphas:pending.put(alpha)
        def next_angles():
            while True:
                try:yield pending.get_nowait()
                except queue.Empty:return
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(worker,next_angles()) for _ in range(min(a.workers,len(alphas)))]
            for future in concurrent.futures.as_completed(futures): future.result()
    rows=plot_results(a.output,alphas)
    print(f'verified {sum(r["converged"] for r in rows)}/{len(rows)}; output={a.output}',flush=True)


if __name__=='__main__': main()
