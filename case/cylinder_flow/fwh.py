"""Two-dimensional convected, frequency-domain surface-only FW-H acoustics.

Time convention Re[p_hat exp(-i omega t)]. Exterior quadrupoles are omitted;
permeable-contour sensitivity is reported, not assumed to vanish.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import hankel1
from scipy.optimize import minimize_scalar


def green(displacement,omega,mean=(0.,0.),c0=1.):
    """Outgoing Green function and observer gradient of the convected wave operator."""
    u=np.asarray(mean,dtype=float);speed=np.linalg.norm(u);mach=speed/c0
    if not 0<=mach<1 or omega<=0:raise ValueError('positive frequency and subsonic mean required')
    direction=u/speed if speed else np.array([1.,0.])
    cross=np.array([-direction[1],direction[0]])
    x=np.asarray(displacement)@direction;y=np.asarray(displacement)@cross
    beta2=1-mach*mach;k=omega/c0
    r=np.sqrt(x*x+beta2*y*y)
    if np.any(r<=0):raise ValueError('observer coincides with source')
    phase=np.exp(-1j*k*mach*x/beta2)*1j/(4*np.sqrt(beta2))
    h0=hankel1(0,k*r/beta2);h1=hankel1(1,k*r/beta2)
    g=phase*h0
    gx=phase*(-1j*k*mach/beta2*h0-k/beta2*h1*x/r)
    gy=-phase*k*h1*y/r
    return g,gx[...,None]*direction+gy[...,None]*cross


def sources(mass_flux,momentum_flux,normals,mean,rho0=1.,p0=1/1.4):
    """Convected FW-H Q and effective F for stationary surfaces.

    Q=J-rho0 U.n; F=Pi.n-p0 n-2 U J+rho0 U(U.n).
    For physical fluxes J=rho*u.n, Pi.n=rho*u*(u.n)+p*n-tau.n.
    The FR export supplies conservative numerical fluxes instead.
    """
    u=np.asarray(mean);un=np.asarray(normals)@u
    q=mass_flux-rho0*un
    f=momentum_flux-p0*normals-2*mass_flux[...,None]*u+rho0*un[...,None]*u
    return q,f


def radiation(points,normals,weights,q,f,observers,omega,mean=(0.,0.),c0=1.):
    g,gradient=green(np.asarray(observers)[:,None,:]-points[None,:,:],omega,mean,c0)
    return np.sum((-1j*omega*g*q[None,:]-np.einsum('osj,sj->os',gradient,f))*weights,axis=-1)


def read_ascii(filename):
    geometry=[]
    with Path(filename).open() as stream:
        first=stream.readline()
        if not first.startswith('# '):raise ValueError('missing ASCII metadata')
        meta=json.loads(first[2:])
        if meta.get('format')!='FR-FWH2D-1':raise ValueError('unsupported ASCII format')
        for line in stream:
            if line.startswith('# point '):geometry.append([float(x) for x in line.split()[2:]])
            elif not line.startswith('#'):break
    geo=np.asarray(geometry);data=np.loadtxt(filename,comments='#',ndmin=2)
    n=len(geo)
    if not n or len(data)%n:raise ValueError('incomplete ASCII sample block')
    if not np.isfinite(geo).all() or np.any(geo[:,6]<=0) or not np.all(geo[:,0]==np.arange(n)):
        raise ValueError('invalid geometry or quadrature weights')
    if not np.allclose(np.linalg.norm(geo[:,4:6],axis=-1),1,atol=1e-10):raise ValueError('invalid unit normals')
    data=data.reshape(-1,n,9)
    if len(data)<16 or not np.isfinite(data).all():raise ValueError('need >=16 finite samples')
    if not np.all(data[:,:,1]==np.arange(n)) or not np.all(data[:,:,0]==data[:,0,0,None]):
        raise ValueError('invalid point order or sample times')
    times=data[:,0,0];dt=np.diff(times)
    if np.any(dt<=0) or not np.allclose(dt,dt.mean(),rtol=1e-7,atol=1e-9):
        raise ValueError('sample times must be uniformly spaced and increasing')
    return meta,geo,times,data


def fit_frequency(times,lift,mach):
    time=(times-times[0])*mach
    centered=lift-lift.mean()
    if np.ptp(centered)<1e-12:raise ValueError('no measurable unsteady lift in sampling window')
    freq=np.fft.rfftfreq(len(time),np.mean(np.diff(time)))
    power=abs(np.fft.rfft(centered*np.hanning(len(time))))**2
    candidates=np.where((freq>=.08)&(freq<=.3))[0]
    if not len(candidates):raise ValueError('record too short to resolve shedding band')
    peak=freq[candidates[np.argmax(power[candidates])]]
    def objective(f):
        phase=2*np.pi*f*time
        matrix=np.column_stack((np.ones(len(time)),time-time.mean(),np.cos(phase),np.sin(phase)))
        difference=lift-matrix@np.linalg.lstsq(matrix,lift,rcond=None)[0]
        return float(difference@difference)
    spacing=freq[1]-freq[0]
    result=minimize_scalar(objective,bounds=(max(.05,peak-spacing),min(.35,peak+spacing)),method='bounded',options={'xatol':1e-12})
    return float(result.x)


def harmonic_fit(times,data,frequency,harmonics):
    time=times-times[0]
    columns=[np.ones(len(time)),time-time.mean()]
    for h in range(1,harmonics+1):columns.extend((np.cos(2*np.pi*h*frequency*time),np.sin(2*np.pi*h*frequency*time)))
    matrix=np.column_stack(columns)
    fit=np.linalg.lstsq(matrix,np.asarray(data).reshape(len(time),-1),rcond=None)[0]
    amplitudes=np.stack([fit[2+2*i]+1j*fit[3+2*i] for i in range(harmonics)])
    return amplitudes.reshape((harmonics,)+data.shape[1:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='ASCII file exported by solve.py --fwh-output')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results/fwh')
    parser.add_argument('--radii',type=float,nargs='+',default=[50.,100.])
    parser.add_argument('--angles',type=int,default=72)
    parser.add_argument('--harmonics',type=int,default=3)
    parser.add_argument('--start',type=float,help='discard samples before this convective time')
    parser.add_argument('--strouhal',type=float,help='prescribe fundamental f D/U_inf instead of fitting lift')
    args=parser.parse_args()
    if args.angles<8 or not 1<=args.harmonics<=8 or any(not math.isfinite(r) or r<=0 for r in args.radii):parser.error('invalid observers/harmonics')
    meta,geo,times,data=read_ascii(args.input)
    if args.start is not None:
        keep=times*meta['mach']>=args.start;times=times[keep];data=data[keep]
        if len(times)<16:raise ValueError('insufficient retained samples')
    labels=geo[:,1].astype(int);points=geo[:,2:4];normals=geo[:,4:6];weights=geo[:,6]
    for radius in args.radii:
        if radius<=np.linalg.norm(points,axis=1).max():raise ValueError('observers must be outside every contour')
    mean=np.asarray(meta['mean_velocity']);mach=meta['mach']
    q,f=sources(data[:,:,6],data[:,:,7:9],normals,mean,meta['rho0'],meta['p0'])
    wall=labels==meta['surface_names'].index('wall')
    lift=np.sum(data[:,wall,8]*weights[wall],axis=1)
    st=fit_frequency(times,lift,mach) if args.strouhal is None else args.strouhal
    if not math.isfinite(st) or st<=0:raise ValueError('positive finite Strouhal number required')
    frequency=st*mach
    if (times[-1]-times[0])*frequency<3:raise ValueError('record must cover at least three fundamental periods')
    if args.harmonics*frequency>=.5/np.mean(np.diff(times)):raise ValueError('harmonic exceeds temporal Nyquist limit')
    qh=harmonic_fit(times,q,frequency,args.harmonics);fh=harmonic_fit(times,f,frequency,args.harmonics)
    theta=np.arange(args.angles)*2*np.pi/args.angles
    radii=np.asarray(args.radii);observers=(radii[:,None,None]*np.stack((np.cos(theta),np.sin(theta)),-1)).reshape(-1,2)
    fields=[]
    for sid,name in enumerate(meta['surface_names']):
        take=labels==sid
        pressure=np.stack([radiation(points[take],normals[take],weights[take],qh[h,take],fh[h,take],
            observers,2*np.pi*frequency*(h+1),mean,meta['c0']) for h in range(args.harmonics)])
        fields.append(pressure)
    pressure=np.asarray(fields).reshape(len(fields),args.harmonics,len(radii),args.angles)
    rms=np.sqrt(np.sum(abs(pressure)**2,axis=1)/2)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for s,name in enumerate(meta['surface_names']):
        for h in range(args.harmonics):
            for r,radius in enumerate(radii):
                for j,angle in enumerate(theta):
                    value=pressure[s,h,r,j]
                    rows.append((s,h+1,radius,np.degrees(angle),abs(value),value.real,value.imag))
    np.savetxt(args.output/'farfield.csv',rows,delimiter=',',header='surface_id,harmonic,r_over_D,angle_deg,pressure_peak,real,imag',comments='')
    report=dict(method='2D convected frequency-domain surface-only FW-H',source=str(args.input),
        surface_names=meta['surface_names'],reynolds=meta['reynolds'],mach=mach,
        strouhal=st,solver_time_frequency=frequency,period_convective=1/st,
        sample_count=len(times),convective_time_range=[times[0]*mach,times[-1]*mach],
        radii=radii.tolist(),angles=args.angles,harmonics=args.harmonics,
        max_rms_by_surface_and_radius=rms.max(axis=-1).tolist(),
        pressure_units='rho_inf*c_inf^2; no dimensional SPL conversion',
        assumptions=['stationary closed contours; uniform subsonic exterior mean',
            'periodic harmonic fit; exterior quadrupole volume term omitted',
            '2D line-source radiation; no finite-span 3D SPL prediction'],
        validation='analytic acoustic tests validate integrals; CFD contour agreement must be assessed separately')
    ref=rms[0].max(axis=-1)
    report['max_rms_ratio_to_wall']=(rms.max(axis=-1)/ref).tolist()
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(args.output,meta,times,lift,st,radii,theta,pressure,rms)
    print(json.dumps(report,indent=2))


def plot(output,meta,times,lift,st,radii,theta,pressure,rms):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(radii),subplot_kw={'projection':'polar'},figsize=(6*len(radii),5),squeeze=False,layout='constrained')
    for r,radius in enumerate(radii):
        ax=axes[0,r]
        for s,name in enumerate(meta['surface_names']):ax.plot(np.r_[theta,theta[0]],np.r_[rms[s,r],rms[s,r,0]],label=name)
        ax.set_title(f'r/D={radius:g}, RMS pressure');ax.legend(loc='upper right')
    fig.savefig(output/'directivity.png',dpi=160);plt.close(fig)
    time=np.linspace(0,3/st,1000)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    observer=int(np.argmin(abs(theta-np.pi/2)))
    traces=[]
    for s,name in enumerate(meta['surface_names']):
        phase=np.exp(-2j*np.pi*st*np.arange(1,pressure.shape[1]+1)[:,None]*time)
        value=np.real(np.sum(pressure[s,:,0,observer,None]*phase,axis=0))
        ax.plot(time,value,label=name);traces.append(value)
    ax.set(xlabel='Periodic receiver time U_inf/D (arbitrary origin)',ylabel="p' / (rho_inf c_inf^2)",title=f'2D FW-H, r/D={radii[0]:g}, theta=90 deg, St={st:.5f}')
    ax.legend();fig.savefig(output/'pressure.png',dpi=160);plt.close(fig)
    np.savetxt(output/'pressure_periodic.csv',np.column_stack((time,*traces)),delimiter=',',header='convective_time,'+','.join(meta['surface_names']),comments='')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for s,name in enumerate(meta['surface_names']):ax.plot(st*np.arange(1,pressure.shape[1]+1),abs(pressure[s,:,0,observer]),'o-',label=name)
    ax.set(xlabel='f D/U_inf',ylabel='Pressure peak amplitude',title='Fitted tonal harmonics (not broadband PSD)');ax.legend()
    fig.savefig(output/'tones.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
