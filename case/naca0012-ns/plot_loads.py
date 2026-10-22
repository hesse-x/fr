"""Plot saved explicit-run convergence and upper/lower wall coefficients."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args(); d=a.directory
    h=np.atleast_1d(np.genfromtxt(d/'history.csv',delimiter=',',names=True))
    residuals=(np.atleast_1d(np.genfromtxt(d/'residuals.csv',delimiter=',',names=True))
               if (d/'residuals.csv').exists() else h)
    w=np.atleast_1d(np.genfromtxt(d/'wall.csv',delimiter=',',names=True))
    stats=json.loads((d/'metrics.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    axes[0,0].semilogy(residuals['step'],residuals['residual'])
    axes[0,0].axhline(stats['residual_tolerance'],color='gray',ls='--')
    axes[0,0].set(xlabel='Explicit step',ylabel='Volume RMS residual')
    recent=h[h['step']>=max(0,h['step'].max()-5000)]
    axes[0,1].plot(recent['step'],recent['CL'],label='CL')
    axes[0,1].plot(recent['step'],recent['CD_total'],label='CD total')
    axes[0,1].set(xlabel='Explicit step (last 5000)',ylabel='Coefficient'); axes[0,1].legend()
    for label,mask in [('upper',w['y']>=0),('lower',w['y']<0)]:
        data=np.sort(w[mask],order='x')
        axes[1,0].plot(data['x'],data['Cp'],label=label)
        axes[1,1].plot(data['x'],data['Cf'],label=label)
    axes[1,0].invert_yaxis()
    axes[1,0].set(xlabel='x/c',ylabel='Cp'); axes[1,1].set(xlabel='x/c',ylabel='Cf (tangent toward +x)')
    for ax in axes.flatten(): ax.grid(alpha=.2)
    axes[1,0].legend(); axes[1,1].legend()
    fig.suptitle(f'Re={stats["reynolds"]:g}, Ma={stats["mach"]:g}, alpha={stats["alpha_deg"]:g} deg; {stats["status"]}')
    fig.tight_layout(); fig.savefig(d/'loads.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
