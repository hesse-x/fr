"""Plot streaming diagnostic files and summarize evidence without declaring convergence."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def plot(folder):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-naca-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder)
    def read(name):return np.atleast_1d(np.genfromtxt(folder/name,delimiter=',',names=True,dtype=None,encoding='utf-8'))
    h=read('history.csv');d=read('diagnostics.csv');b=read('boundary.csv');p=read('probes.csv')
    m=json.loads((folder/'metrics.json').read_text())
    # Align live CSV streams with the last complete checkpoint/history save.
    d=d[d['iteration']<=h['iteration'][-1]]
    b=b[b['iteration']<=h['iteration'][-1]]
    p=p[p['iteration']<=h['iteration'][-1]]
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    axes[0,0].plot(d['time'],np.interp(d['iteration'],h['iteration'],h['CL']),label='CL')
    axes[0,0].plot(d['time'],np.interp(d['iteration'],h['iteration'],h['CD_pressure']),label='CD pressure')
    axes[0,0].legend();axes[0,0].set_title('Loads')
    for name in ['residual_near','residual_wake','residual_outer']:
        axes[0,1].semilogy(d['time'],np.maximum(d[name],1e-18),label=name)
    axes[0,1].legend();axes[0,1].set_title('Regional residuals (physical quadrature RMS)')
    for name in ['upper_near','lower_near','wake_2c','wake_5c']:
        mask=p['name']==name;axes[0,2].plot(p['time'][mask],p['p_prime'][mask],label=name)
    axes[0,2].legend();axes[0,2].set_title('Near-field pressure probes')
    for side in ['left','right','top','bottom']:
        mask=b['side']==side
        axes[1,0].semilogy(b['time'][mask],np.maximum(b['p_prime_rms'][mask],1e-18),label=side)
    axes[1,0].legend();axes[1,0].set_title('Far boundary pressure perturbation RMS')
    axes[1,1].semilogy(d['time'],np.maximum(d['wall_un_rms'],1e-18),label='wall normal velocity RMS')
    axes[1,1].semilogy(d['time'],np.maximum(d['wall_un_max'],1e-18),label='wall normal velocity max')
    axes[1,1].legend();axes[1,1].set_title('Slip trace error')
    axes[1,2].semilogy(d['time'],np.maximum(np.abs(d['mass_balance_error']),1e-18),label='mass')
    axes[1,2].semilogy(d['time'],np.maximum(np.abs(d['energy_balance_error']),1e-18),label='energy')
    axes[1,2].legend();axes[1,2].set_title('Instantaneous integral(rhs) + boundary flux')
    for ax in axes.flat:
        ax.set_xlabel('Physical time (c/a_inf)' if m['time_mode']=='physical_time' else 'Minimum accumulated pseudo time')
        ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(folder/'diagnostics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for side in ['left','right','top','bottom']:
        mask=b['side']==side
        for ax,key in zip(axes,['acoustic_out_rms','acoustic_in_rms']):
            ax.semilogy(b['time'][mask],np.maximum(b[key][mask],1e-18),label=side)
    for ax,title in zip(axes,['Outgoing acoustic indicator','Incoming acoustic indicator']):
        ax.set_title(title);ax.set_xlabel('Time');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'boundary_acoustics.png',dpi=180);plt.close(fig)
    files=sorted(path for path in (folder/'snapshots').glob('cells_*.csv')
                 if int(path.stem.split('_')[-1])<=h['iteration'][-1])
    if files:
        c=np.genfromtxt(files[-1],delimiter=',',names=True)
        import matplotlib.colors as colors
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        pscale=max(float(np.max(np.abs(c['p']-1/1.4))),1e-10)
        for ax,values,title,norm in [
            (axes[0],c['p']-1/1.4,'Pressure perturbation',colors.TwoSlopeNorm(vmin=-pscale,vcenter=0,vmax=pscale)),
            (axes[1],np.maximum(c['residual_rms_max'],1e-18),'Cell residual RMS max',colors.LogNorm()),
            (axes[2],c['entropy']-1/1.4,'Entropy proxy p/rho^gamma - freestream',None)]:
            art=ax.scatter(c['x'],c['y'],c=values,s=7,cmap='coolwarm' if 'Pressure' in title else 'viridis',norm=norm)
            ax.set_xlim(-1,3);ax.set_ylim(-1.5,1.5);ax.set_aspect('equal');ax.set_title(title);fig.colorbar(art,ax=ax)
        fig.tight_layout();fig.savefig(folder/'last_field.png',dpi=180);plt.close(fig)
    report=dict(status=m['status'],steps=int(h['iteration'][-1]),time=float(d['time'][-1]),time_mode=m['time_mode'],
                converged=m['steady_convergence_verified'],CL=float(h['CL'][-1]),CD_pressure=float(h['CD_pressure'][-1]),
                residual_initial=float(h['residual_rms_max'][0]),residual_final=float(h['residual_rms_max'][-1]),
                maximum_far_boundary_pressure_rms=float(np.max(b['p_prime_rms'])),
                maximum_mass_balance_error=float(np.max(np.abs(d['mass_balance_error']))),
                maximum_energy_balance_error=float(np.max(np.abs(d['energy_balance_error']))),
                wall_normal_velocity_rms_initial=float(d['wall_un_rms'][0]),wall_normal_velocity_rms_final=float(d['wall_un_rms'][-1]),
                residual_region_final={key:float(d[key][-1]) for key in ['residual_near','residual_wake','residual_outer']},
                caveats=['Acoustic amplitudes are local normal linearized indicators, not measured reflection coefficients.',
                         'Probe values are fixed nearest solution DOFs; actual positions are in probes.json.',
                         'A finite-duration run cannot establish asymptotic convergence or rule out later reflections.'])
    (folder/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args();plot(a.folder)
