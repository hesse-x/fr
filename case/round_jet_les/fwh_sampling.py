"""Uniform-time samples on stationary permeable FW–H surfaces for the round jet."""
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def _gauss_panels(lo, hi, max_width):
    edges = np.linspace(lo, hi, math.ceil((hi-lo)/max_width)+1)
    center, half = (edges[1:]+edges[:-1])/2, np.diff(edges)/2
    z = np.array([-1., 1.])/math.sqrt(3.)
    return (center[:, None]+half[:, None]*z).ravel(), np.repeat(half, 2)


def surface_quadrature(plan, *, azimuth_count=192, panel_width=.4):
    """Build two side surfaces and reuse patches between alternative closures."""
    start = float(plan['upstream_cap_x_D'])
    end = float(plan['nominal_downstream_cap_x_D'])
    ends = sorted({end, *map(float, plan['alternative_downstream_caps_x_D'])})
    a, b, c = map(float, plan['radius_coefficients'])
    offset = float(plan['secondary_surface_radius_offset_D'])
    if not 0 <= start < ends[0] or offset <= 0 or azimuth_count < 4 or panel_width <= 0:
        raise ValueError('invalid FW–H surface bounds or quadrature')
    theta = np.arange(azimuth_count)*2*np.pi/azimuth_count
    cos, sin = np.cos(theta), np.sin(theta)
    dtheta = 2*np.pi/azimuth_count
    coordinates, normals, weights = [], [], []
    surfaces, patches = {}, {}
    start_idx = 0

    def append_patch(name, x, radius, nx, nr, area):
        x, radius, nx, nr, area = [np.asarray(v)[:, None] for v in (x, radius, nx, nr, area)]
        xyz = np.stack(np.broadcast_arrays(x, radius*cos, radius*sin), axis=-1).reshape(-1, 3)
        normal = np.stack(np.broadcast_arrays(nx, nr*cos, nr*sin), axis=-1).reshape(-1, 3)
        weight = np.broadcast_to(area*dtheta, (len(x), azimuth_count)).ravel()
        nonlocal start_idx
        patches[name] = [start_idx, start_idx+len(xyz)]
        start_idx += len(xyz)
        coordinates.append(xyz)
        normals.append(normal)
        weights.append(weight)
        return name

    for kind, radial_offset, caps in (('nominal', 0., ends), ('wide', offset, [end])):
        def radius(x):
            s = x-start
            return a+b*s+c*s*s+radial_offset

        def disk(x, sign):
            r, dr = _gauss_panels(0., radius(x), panel_width)
            return append_patch(f'{kind}_disk_{x:g}', np.full_like(r, x), r,
                                np.full_like(r, sign), np.zeros_like(r), r*dr)

        patch_names = [disk(start, -1.)]
        left = start
        for right in caps:
            x, dx = _gauss_panels(left, right, panel_width)
            slope = b+2*c*(x-start)
            length = np.hypot(1., slope)
            r = radius(x)
            if np.any(r <= 0.):
                raise ValueError('FW–H side radius must be positive')
            patch_names.append(append_patch(f'{kind}_side_{left:g}_{right:g}', x, r,
                                           -slope/length, 1./length, r*length*dx))
            surfaces[f'{kind}_{right:g}D'] = [*patch_names, disk(right, 1.)]
            left = right
    xyz, normal, area = map(np.concatenate, (coordinates, normals, weights))
    if not all(np.isfinite(v).all() for v in (xyz, normal, area)):
        raise ValueError('nonfinite FW–H geometry')
    for names in surfaces.values():
        vector_area = sum((normal[lo:hi]*area[lo:hi,None]).sum(axis=0)
                          for lo, hi in (patches[name] for name in names))
        if np.linalg.norm(vector_area) > 1e-9:
            raise ValueError('FW–H surface fails the closed-area check')
    return xyz, normal, area, patches, surfaces


class FWHRecorder:
    def __init__(self, model, directory, plan_path, interval, time, committed):
        if not math.isfinite(interval) or interval <= 0.:
            raise ValueError('FW–H sampling interval must be positive and finite')
        self.interval = interval
        self.count = 0 if committed is None else committed
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory/'metadata.json'
        geometry_path = directory/'quadrature.npz'
        data_path = directory/'surface.f32'
        plan = json.loads(Path(plan_path).read_text())
        xyz, normal, area, patches, surfaces = surface_quadrature(plan)
        if xyz[:,0].max() >= model.edges[-1] or xyz[:,0].min() <= 0.:
            raise ValueError('FW–H endcaps must lie strictly inside the axial domain')
        self.point_count = len(xyz)
        self.origin = time if not meta_path.exists() else json.loads(meta_path.read_text())['start_time_Uj_D']
        meta = dict(format='FR-FWH3D-1', mesh_sha256=model.mesh_digest,
                    start_time_Uj_D=self.origin, interval_Uj_D=interval, point_count=len(xyz),
                    dtype='<f4', columns=['rho_minus_1', 'u', 'v', 'w', 'p_minus_p0'],
                    rho0=1., p0=model.pressure, c0=1/model.mach,
                    patches=patches, surfaces=surfaces, plan=plan)
        self.model = model
        self.evaluate = model.probes(tuple(torch.from_numpy(xyz[:, i].copy()) for i in range(3)))
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != meta:
                raise ValueError('FW–H restart configuration differs')
            with np.load(geometry_path) as geometry:
                if not (np.array_equal(geometry['xyz_D'], xyz) and
                        np.array_equal(geometry['outward_normal'], normal) and
                        np.array_equal(geometry['area_D2'], area)):
                    raise ValueError('FW–H restart quadrature differs')
        else:
            if self.count or data_path.exists():
                raise ValueError('missing FW–H metadata for existing samples')
            np.savez(geometry_path, xyz_D=xyz, outward_normal=normal, area_D2=area)
            temp = meta_path.with_suffix('.tmp')
            temp.write_text(json.dumps(meta,indent=2)+'\n')
            temp.replace(meta_path)
        if self.count:
            if self.origin+(self.count-1)*interval > time+1e-10 or self.next_time < time-1e-10:
                raise ValueError('FW–H sample count is inconsistent with the checkpoint time')
        elif abs(self.origin-time)>1e-10:
            raise ValueError('FW–H recording can only start at its original checkpoint')
        self.stream = data_path.open('a+b')
        size = self.count*self.point_count*5*4
        if self.stream.tell() < size:
            self.stream.close()
            raise ValueError('FW–H data shorter than the committed sample count')
        self.stream.truncate(size)

    @property
    def next_time(self):
        return self.origin+self.interval*self.count

    def record(self, state, time, previous=None, previous_time=None):
        if time+1e-12 < self.next_time:
            return
        current = self.evaluate(state)
        earlier = current if previous is None else self.evaluate(previous)
        while self.next_time <= time+1e-12:
            if previous is None:
                values = current
            else:
                fraction = (self.next_time-previous_time)/(time-previous_time)
                values = torch.lerp(earlier, current, min(1., max(0., fraction)))
            rho, velocity, p = self.model.eq.primitive(values)
            if not torch.isfinite(values).all() or torch.any(rho<=0.) or torch.any(p<=0.):
                raise RuntimeError('nonphysical interpolated FW–H state')
            block = torch.cat(((rho-1.)[:,None], velocity, (p-self.model.pressure)[:,None]), dim=-1)
            self.stream.write(block.numpy().astype('<f4').tobytes())
            self.count += 1

    def sync(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()
