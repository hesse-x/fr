"""Time the mapped round-jet kernel on a saved checkpoint."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from .mapped import MappedRoundJetLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--group-packet', type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, mmap=True, map_location='cpu', weights_only=True)
    config = saved['config']
    mesh = args.mesh or args.checkpoint.with_name('mesh.msh')
    model = MappedRoundJetLES(mesh, reynolds=1/config['mu'], mach=config['mach'],
                             shear_width=config['shear_width'], disturbance=config['disturbance'],
                             modal_filter=config['modal_filter'],
                             outlet=config.get('outlet', 'characteristic'),
                             initial_flow=config.get('initial_flow', 'rest'))
    if model.mesh_digest != config['hex_mesh_sha256']:
        raise ValueError('mesh does not match checkpoint')
    options = {} if args.group_packet is None else {'group_packet': args.group_packet}
    solver = model.make_stepper('native-hex-p3', **options)
    state, t, cfl = saved['state'], saved['time'], config['cfl']
    valid, limit = solver.scan(state, cfl, t=t)
    if not valid:
        raise RuntimeError('invalid checkpoint state')
    dt = .8*limit
    output = torch.empty_like(state)
    timings, pending_work = [], []
    for sample in range(12):
        begin = perf_counter()
        if not solver.scan(state, cfl, t=t)[0]:
            raise RuntimeError('scan failed')
        scanned = perf_counter()
        if not solver.advance_finish(state, t, dt, cfl, out=output)[1]:
            raise RuntimeError('step failed')
        end = perf_counter()
        if sample >= 3:
            timings.append(((scanned-begin)*1000, (end-scanned)*1000))
            pending_work.append(solver.deferred_work())

    scan, advance = map(median, zip(*timings))
    faces, cells = map(median, zip(*pending_work))
    meta = solver.metadata
    print(f'cells={state.shape[0]} step={saved["step"]} dt={dt:.10g} '
          f'threads={args.threads} group_packet={meta["group_packet"]}')
    print(f'scan={scan:.3f} ms, advance_finish={advance:.3f} ms, '
          f'pending_faces={faces}, pending_cells={cells}, '
          f'buffers_mib={meta["deferred_buffer_bytes"]/2**20:.1f}, '
          f'within_packet={meta["interface_chunk_fraction"]:.1%}')


if __name__ == '__main__':
    main()
