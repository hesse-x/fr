"""Stationary, three-dimensional permeable FW–H surface projection."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import fft, integrate, signal


def _modal_sources(data, normals, modes, taper_samples, p0):
    """Project the physical mass and loading flux into cylindrical Fourier modes."""
    count, rings, azimuth, _ = data.shape
    result = np.empty((4, rings, modes+1, count), dtype=np.complex64)
    theta = 2*np.pi*np.arange(azimuth, dtype=np.float32)/azimuth
    cos, sin = np.cos(theta), np.sin(theta)
    normal = normals.astype(np.float32)
    nx = normal[...,0]
    nr = normal[...,1]*cos+normal[...,2]*sin
    for start in range(0, count, 32):
        stop = min(start+32, count)
        rho, u, v, w, pressure = np.moveaxis(data[start:stop],-1,0)
        radial, tangent = cos*v+sin*w, cos*w-sin*v
        mass = (1+rho)*(nx*u+nr*radial)
        sources = np.stack((mass,mass*u+pressure*nx,mass*radial+pressure*nr,mass*tangent))
        if not np.isfinite(sources).all() or np.any(1+rho<=0) or np.any(p0+pressure<=0):
            raise ValueError('nonphysical FW–H surface data')
        spectrum = fft.rfft(sources,axis=-1)
        result[...,start:stop] = spectrum[...,:modes+1].transpose(0,2,3,1)/azimuth
    taper = signal.windows.tukey(count,alpha=min(1.,2*taper_samples/(count-1))).astype(np.float32)
    for block in result:
        block -= block.mean(axis=-1,keepdims=True)
        block *= taper
    return result


def _time_transform(source, nfft, count):
    shape = (*source.shape[:-2], 2*source.shape[-2]-1, count)
    out = np.empty(shape,dtype=np.complex64)
    for variable in range(source.shape[0]):
        for start in range(0,source.shape[1],8):
            ring = source[variable,start:start+8]
            transform = fft.fft(ring,n=nfft,axis=-1,workers=2)
            positive, negative = out[variable,start:start+len(ring)], np.arange(count)
            positive[:,:ring.shape[-2]] = transform[...,:count]
            positive[:,ring.shape[-2]:] = transform[:,1:,(-negative)%nfft].conj()
    return out.transpose(3,1,2,0).copy()


def _observer_kernel(rings_x, rings_r, weight, azimuth, polar, distance, frequency, sound, modes):
    """Angular Green kernels with the common travel time distance/c0 removed."""
    theta = 2*np.pi*np.arange(azimuth)/azimuth
    x = distance*np.cos(polar)-rings_x[:,None]
    y = distance*np.sin(polar)
    radial = y*np.cos(theta)-rings_r[:,None]
    tangent = -y*np.sin(theta)
    distance_sq = x*x+radial*radial+tangent*tangent
    separation = np.sqrt(distance_sq)
    delay = ((separation-distance)/sound).astype(np.float32)
    scale = (weight[:,None]/(4*np.pi*separation)).astype(np.float32)
    wave = (2j*np.pi*frequency).astype(np.complex64)[:,None,None]
    green = np.exp(-wave*delay)*scale
    inverse = np.reciprocal(separation.astype(np.float32))
    derivative = (wave/sound+inverse)*inverse*green
    mode_indices = np.r_[0,-np.arange(1,modes+1),np.arange(1,modes+1)]
    monopole = fft.fft(wave*green,axis=-1,workers=4)[...,mode_indices%azimuth]
    dipole = fft.fft(derivative,axis=-1,workers=4)
    mid, prev, nxt = (dipole[...,ind%azimuth] for ind in
                      (mode_indices,mode_indices-1,mode_indices+1))
    # cos and sin projections shift the Fourier index by one.
    loading_x = mid*x.astype(np.float32)
    loading_r = .5*y*(prev+nxt)-mid*rings_r.astype(np.float32)[:,None]
    loading_theta = .5j*y*(prev-nxt)
    return np.stack((monopole,loading_x,loading_r,loading_theta),axis=-1).astype(np.complex64)


def _ring_indices(meta, azimuth):
    return np.array([
        [int(any(meta['patches'][patch][0] <= j < meta['patches'][patch][1] for patch in names))
         for j in range(0,meta['point_count'],azimuth)]
        for names in meta['surfaces'].values()],dtype=np.float32)


def _project_observer(source, omega, ring_geometry, polar, distance, sound, modes, masks):
    out = np.empty((len(masks),len(omega),source.shape[2]),dtype=np.complex64)
    for first in range(0,len(omega),12):
        part = slice(first,first+12)
        kernel = _observer_kernel(*ring_geometry,polar,distance,omega[part],sound,modes)
        ring_pressure = (source[part]*kernel).sum(-1)
        out[:,part] = np.einsum('sr,frm->sfm',masks,ring_pressure)
    return out


def project(directory, *, output, radii=(50.,100.), polar_count=37, azimuth_count=96,
            modes=36, band=(.05,2.), window=100., max_frames=None):
    meta = json.loads((directory/'metadata.json').read_text())
    if meta['format'] != 'FR-FWH3D-1':
        raise ValueError('unrecognized FW–H input')
    geometry = np.load(directory/'quadrature.npz')
    xyz, normal, area = (geometry[name] for name in ('xyz_D','outward_normal','area_D2'))
    first_patch = next(iter(meta['patches'].values()))
    # A ring has constant x and r. Recover the ring length from its Cartesian angles.
    angle = np.unwrap(np.arctan2(xyz[first_patch[0]:first_patch[1],2],
                                 xyz[first_patch[0]:first_patch[1],1]))
    azimuth = int(round(2*np.pi/(angle[1]-angle[0])))
    if not 0 <= modes < azimuth//2 or azimuth_count < 2*modes+1 or polar_count < 2:
        raise ValueError('invalid projection angles or number of azimuthal modes')
    ring_count = meta['point_count']//azimuth
    xyz, normal, area = (a.reshape(ring_count,azimuth,*a.shape[1:]) for a in (xyz,normal,area))
    radius = np.linalg.norm(xyz[...,1:],axis=-1)
    for quantity in (xyz[...,0],radius,area):
        if not np.allclose(quantity,quantity[:,[0]],rtol=1e-12,atol=1e-12):
            raise ValueError('FW–H input is not a rotational ring quadrature')
    data_path = directory/'surface.f32'
    count, leftover = divmod(data_path.stat().st_size,meta['point_count']*5*4)
    if max_frames is not None:
        if not 0 < max_frames <= count:
            raise ValueError('requested FW–H frames are not available')
        count = max_frames
    elif leftover:
        raise ValueError('incomplete FW–H input frame')
    dt, sound = meta['interval_Uj_D'],meta['c0']
    source_extent = np.linalg.norm(xyz,axis=-1).max()
    if min(radii) <= source_extent or not 0 < band[0] < band[1] < .4/dt:
        raise ValueError('observation sphere or analysis frequency band is invalid')
    guard = math.ceil(source_extent/(sound*dt))
    taper_samples = max(4,min(guard//4,round(2./dt)))
    trim = guard+taper_samples
    nperseg = round(window/dt)
    length = count-2*trim
    if nperseg<16 or length<nperseg:
        raise ValueError('FW–H recording is too short for the requested spectral window')
    nfft = fft.next_fast_len(count+2*guard)
    frequency = fft.rfftfreq(nfft,dt)
    passband_end = min(1.2*band[1], .5/dt)
    nfreq = np.searchsorted(frequency,passband_end,side='right')
    rolloff = .5*(1+np.cos(np.pi*np.clip((frequency[:nfreq]-band[1])/(passband_end-band[1]),0.,1.)))
    data = np.memmap(data_path,dtype=meta['dtype'],mode='r',shape=(count,ring_count,azimuth,5))
    output.mkdir(parents=True,exist_ok=True)
    print(f'FW–H: {count} source frames, {ring_count} rings, {nfreq} frequency bins',flush=True)
    modal = _modal_sources(data,normal,modes,taper_samples,meta['p0'])
    source = _time_transform(modal,nfft,nfreq)
    del modal
    rings = xyz[:,0,0],radius[:,0],area[:,0],azimuth
    masks = _ring_indices(meta,azimuth)
    meridian = np.linspace(0.,np.pi,polar_count)
    longitude = np.arange(azimuth_count)*2*np.pi/azimuth_count
    spectrum_axis = fft.rfftfreq(nperseg,dt)
    psd = np.empty((len(masks),len(radii),polar_count,azimuth_count,len(spectrum_axis)),
                   dtype=np.float32)
    mode_idx = np.r_[0,np.arange(1,modes+1),(-np.arange(1,modes+1))%azimuth_count]
    for ridx,distance in enumerate(radii):
        for tidx,polar in enumerate(meridian):
            modal_pressure = _project_observer(source,frequency[:nfreq],rings,polar,
                                               distance,sound,modes,masks)
            spectral = np.zeros((len(masks),nfreq,azimuth_count),dtype=np.complex64)
            spectral[...,mode_idx] = modal_pressure*rolloff[None,:,None]
            pressure_hat = fft.ifft(spectral,axis=-1,workers=2)*azimuth_count
            pressure = fft.irfft(pressure_hat,n=nfft,axis=1,workers=2)[:,trim:count-trim]
            _, psd[:,ridx,tidx] = signal.welch(pressure.transpose(0,2,1),fs=1/dt,
                                                nperseg=nperseg,noverlap=nperseg//2)
            if tidx % 6 == 0 or tidx==len(meridian)-1:
                print(f'FW–H: R={distance:g}D, polar angle {np.rad2deg(polar):g}°',flush=True)
    retained = spectrum_axis<=band[1]
    spectrum_axis, psd = spectrum_axis[retained], psd[...,retained]
    selection = spectrum_axis>=band[0]
    if np.count_nonzero(selection)<2:
        raise ValueError('too few spectral bins in the analysis range')
    rms = np.sqrt(integrate.trapezoid(psd[...,selection],spectrum_axis[selection],axis=-1))
    np.savez_compressed(output/'spherical_spectra.npz',St=spectrum_axis,
                        psd_p2_per_St=psd,surface=np.array(list(meta['surfaces'])),
                        observer_radius_D=radii,polar_deg=np.rad2deg(meridian),
                        azimuth_deg=np.rad2deg(longitude),band=band,rms=rms)
    result = dict(input_frames=count,pressure_samples=length,surface=list(meta['surfaces']),
                  start_time_Uj_D=meta['start_time_Uj_D'],sample_interval_Uj_D=dt,
                  common_duration_Uj_D=(length-1)*dt,frequency_window_Uj_D=nperseg*dt,
                  welch_segments=1+(length-nperseg)//(nperseg-nperseg//2),
                  band_St=list(band),azimuth_modes=modes,
                  pressure_units='rho_inf Uj^2',observer_angle='0 deg downstream')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    _plot(output,radii,meridian,longitude,list(meta['surfaces']),rms,band)
    return result


def _plot(output,radii,polar,azimuth,surfaces,rms,band):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nominal = surfaces.index('nominal_22D') if 'nominal_22D' in surfaces else 0
    reference = np.maximum(rms,np.finfo(rms.dtype).tiny)
    avg = np.sqrt(np.mean(reference**2,axis=-1))
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.6,4.1),layout='constrained')
    theta = np.rad2deg(polar)
    for j,distance in enumerate(radii):
        ax.plot(theta,20*np.log10(avg[nominal,j]*distance),label=f'{distance:g}D')
    ax.set(xlabel='Polar angle (deg)',ylabel='Distance-scaled sound level (dB re nondimensional unit)')
    ax.legend(frameon=False)
    last = len(radii)-1
    for i,name in enumerate(surfaces):
        bx.plot(theta,20*np.log10(avg[i,last]/avg[nominal,last]),label=name)
    bx.set(xlabel='Polar angle (deg)',ylabel=f'Surface sensitivity at {radii[last]:g}D (dB)')
    for axis in (ax,bx):
        axis.grid(alpha=.25)
    bx.legend(frameon=False)
    fig.suptitle(f'FW–H directivity · St={band[0]:g}–{band[1]:g}')
    fig.savefig(output/'directivity.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.1,4.),layout='constrained')
    field=20*np.log10(reference[nominal,last])
    image=ax.pcolormesh(np.r_[np.rad2deg(azimuth),360.],theta,np.c_[field,field[:,0]],
                       shading='auto',cmap='viridis')
    ax.set(xlabel='Azimuth angle (deg)',ylabel='Polar angle (deg)',
           title=f'Spherical sound level at {radii[last]:g}D · St={band[0]:g}–{band[1]:g}')
    fig.colorbar(image,ax=ax,label='dB re nondimensional pressure unit')
    fig.savefig(output/'spherical_sound_level.png',dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='directory containing the FW–H recorder output')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--radii',type=float,nargs='+',default=[50.,100.])
    parser.add_argument('--polar-count',type=int,default=37)
    parser.add_argument('--azimuth-count',type=int,default=96)
    parser.add_argument('--modes',type=int,default=36)
    parser.add_argument('--band',type=float,nargs=2,default=[.05,2.])
    parser.add_argument('--window',type=float,default=100.)
    parser.add_argument('--max-frames',type=int,help='read only this many committed frames for a preview')
    args=parser.parse_args()
    project(args.directory,output=args.output or args.directory/'farfield',radii=args.radii,
            polar_count=args.polar_count,azimuth_count=args.azimuth_count,modes=args.modes,
            band=args.band,window=args.window,max_frames=args.max_frames)


if __name__=='__main__':
    main()
