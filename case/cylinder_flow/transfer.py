"""Quadrature L2 transfer of a checkpoint between curved cylinder meshes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from cylinder import Cylinder


def sample_curved(source,U,points,chunk=4096):
    """Locate physical points by curved-map inversion, then evaluate cell state."""
    b=source.s.blocks[0]; X=b['X']; geometry=b['geometry']
    quad=b['basis'].kind=='quad'
    centers=X[:,:4 if quad else 3].mean(1)
    values=U.reshape(b['ne'],b['ns'],4)
    sampled=[]
    for part in points.split(chunk):
        # No optional spatial-index dependency; bounded temporary distance matrix.
        distances=torch.cdist(part,centers)
        found=torch.zeros(len(part),dtype=torch.bool)
        result=part.new_empty(len(part),4)
        for count in (min(32,len(centers)),len(centers)):
            candidates=distances.topk(count,largest=False).indices
            for rank in range(count):
                pending=torch.where(~found)[0]
                if not len(pending): break
                ids=candidates[pending,rank]; cells=X[ids]; xy=part[pending]
                matrix=torch.stack((cells[:,1]-cells[:,0],cells[:,3 if quad else 2]-cells[:,0]),-1)
                ref=torch.linalg.solve(matrix,(xy-cells[:,0])[...,None])[...,0]
                for _ in range(6):
                    mapped=torch.einsum('qi,qik->qk',geometry.evaluate(ref),cells)
                    xr=torch.einsum('qi,qik->qk',geometry.evaluate(ref,0),cells)
                    xs=torch.einsum('qi,qik->qk',geometry.evaluate(ref,1),cells)
                    residual=mapped-xy
                    det=xr[:,0]*xs[:,1]-xr[:,1]*xs[:,0]
                    safe=torch.where(det.abs()>1e-14,det,torch.ones_like(det))
                    delta=torch.stack((xs[:,1]*residual[:,0]-xs[:,0]*residual[:,1],
                                       -xr[:,1]*residual[:,0]+xr[:,0]*residual[:,1]),-1)/safe[:,None]
                    ref-=delta
                error=(torch.einsum('qi,qik->qk',geometry.evaluate(ref),cells)-xy).norm(dim=-1)
                upper=ref.max(-1).values if quad else ref.sum(-1)
                inside=(ref.min(-1).values>=-1e-7)&(upper<=1+1e-7)&(error<2e-9)
                accepted=pending[inside]
                result[accepted]=torch.einsum('qi,qiv->qv',b['basis'].evaluate(ref[inside]),values[ids[inside]])
                found[accepted]=True
            if found.all(): break
        if not found.all():
            raise ValueError(f'{int((~found).sum())} target points outside the source curved mesh')
        sampled.append(result)
    return torch.cat(sampled)


def project(source,target,U):
    b=target.s.blocks[0]
    samples=sample_curved(source,U,b['xy'].reshape(-1,2)).reshape(b['ne'],-1,4)
    right=torch.einsum('qi,eq,eqv->eiv',b['B'],b['weight'],samples)
    return torch.cholesky_solve(right,b['chol']).reshape(-1,4)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-mesh',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,weights_only=False); old=checkpoint['stats']
    if old['mesh_sha256']!=hashlib.sha256(args.source_mesh.read_bytes()).hexdigest():
        raise ValueError('source mesh does not match checkpoint')
    if old['time_mode']!='global': raise ValueError('transfer requires a physical-time checkpoint')
    kwargs=dict(p=old['p'],quadrature_order=old['quadrature'],mach=old['mach'],
                alpha=old['alpha_deg'],reynolds=old['reynolds'])
    source=Cylinder(args.source_mesh,**kwargs); target=Cylinder(args.mesh,**kwargs)
    U=project(source,target,checkpoint['U'])
    if not target.valid(U): raise ValueError('projected state fails density/pressure checks')
    before=source.s.integral(checkpoint['U']); after=target.s.integral(U)
    report=dict(source_mesh=str(args.source_mesh),target_mesh=str(args.mesh),
                source_checkpoint=str(args.checkpoint),convective_time=old['convective_time'],
                source_cells=source.s.blocks[0]['ne'],target_cells=target.s.blocks[0]['ne'],
                integral_before=before.tolist(),integral_after=after.tolist(),
                integral_difference=(after-before).tolist(),
                method='curved-cell point location and quadrature L2 projection; no spatial smoothing',
                note='New-grid step counter and load history reset; physical time preserved. Further evolution required before recording.')
    stats=dict(old,steps=0,cells=target.s.blocks[0]['ne'],ndof=target.s.ndof,
               mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),status='mesh_transfer')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(U=U,coords=target.s.coords,stats=stats,history=[]),args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
