"""Corner-shape and sampled high-order geometry quality of triangular meshes."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fr import Mesh2D
from fr.reference import Basis,gmsh_reference_nodes,quadrature


def evaluate(mesh_path,output):
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    mesh=Mesh2D.read(mesh_path);rows=[];offset=0
    # Dense reference lattice includes vertices and curved edges, supplemented
    # with integration points. This samples positivity; it is not a certificate.
    lattice=torch.tensor([(i/24,j/24) for j in range(25) for i in range(25-j)])
    q,qw=quadrature('tri',8)
    check=torch.cat((lattice,q))
    for b in mesh.blocks:
        if b.kind!='tri':raise ValueError('triangle-only quality tool')
        xyz=b.geometry[:,:3]
        edges=xyz.roll(-1,1)-xyz
        lens=edges.norm(dim=-1)
        cross=edges[:,0,0]*(-edges[:,2,1])-edges[:,0,1]*(-edges[:,2,0])
        area=.5*cross.abs()
        shape=4*math.sqrt(3)*area/lens.square().sum(1)
        cosine=(-(edges*edges.roll(1,1)).sum(-1)/(lens*lens.roll(1,1))).clamp(-1,1)
        angles=torch.acos(cosine)*180/math.pi
        # Longest-edge / corresponding altitude, normalized so equilateral=1.
        aspect=math.sqrt(3)/2*lens.amax(1).square()/(2*area)
        g=Basis('tri',b.order,nodes=gmsh_reference_nodes('tri',b.order))
        dr=g.evaluate(check,0);ds=g.evaluate(check,1)
        xr=torch.einsum('qi,eik->eqk',dr,b.geometry)
        xs=torch.einsum('qi,eik->eqk',ds,b.geometry)
        J=xr[...,0]*xs[...,1]-xr[...,1]*xs[...,0]
        scaled=J/(xr.norm(dim=-1)*xs.norm(dim=-1))
        qr=torch.einsum('qi,eik->eqk',g.evaluate(q,0),b.geometry)
        qs=torch.einsum('qi,eik->eqk',g.evaluate(q,1),b.geometry)
        jq=qr[...,0]*qs[...,1]-qr[...,1]*qs[...,0]
        xy=torch.einsum('qi,eik->eqk',g.evaluate(q),b.geometry)
        physical_area=(jq*qw).sum(1)
        center=(xy*(jq*qw)[...,None]).sum(1)/physical_area[:,None]
        vals=torch.stack((torch.arange(offset,offset+len(xyz)),center[:,0],center[:,1],area,physical_area,
                          shape,angles.amin(1),angles.amax(1),aspect,lens.amin(1),lens.amax(1),
                          J.amin(1),J.amax(1),J.amin(1)/J.amax(1),scaled.amin(1)),1)
        rows.append(vals.numpy());offset+=len(xyz)
    data=np.concatenate(rows)
    names=['cell_id','x','y','corner_area','curved_area','shape_quality','min_angle_deg','max_angle_deg',
           'aspect_normalized','edge_min','edge_max','jacobian_min','jacobian_max','jacobian_min_max_ratio','scaled_jacobian_min']
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savetxt(output/'cells.csv',data,delimiter=',',header=','.join(names),comments='')
    def stats(v):return dict(min=float(v.min()),p01=float(np.percentile(v,1)),p05=float(np.percentile(v,5)),
                            median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max()))
    metrics={name:stats(data[:,i]) for i,name in enumerate(names) if i>=3}
    face_cell=torch.empty(mesh.nfaces,dtype=torch.long)
    start=0
    for b,ids in zip(mesh.blocks,mesh.face_ids):
        face_cell[ids]=torch.arange(start,start+len(b.geometry))[:,None].expand_as(ids)
        start+=len(b.geometry)
    face_ids=torch.arange(mesh.nfaces)
    interior=(~mesh.boundary)&(face_ids<mesh.neighbor)
    left=face_cell[interior].numpy();right=face_cell[mesh.neighbor[interior]].numpy()
    adjacent_ratio=np.sqrt(np.maximum(data[left,4]/data[right,4],data[right,4]/data[left,4]))
    metrics['adjacent_sqrt_area_ratio']=stats(adjacent_ratio)
    report=dict(mesh=str(Path(mesh_path).resolve()),cells=len(data),metrics=metrics,
                sample_points_per_cell=len(check),
                counts=dict(sampled_nonpositive_jacobian=int((data[:,11]<=0).sum()),
                            min_angle_below_10=int((data[:,6]<10).sum()),min_angle_below_20=int((data[:,6]<20).sum()),
                            quality_below_point2=int((data[:,5]<.2).sum()),quality_below_point5=int((data[:,5]<.5).sum()),
                            adjacent_size_ratio_above_2=int((adjacent_ratio>2).sum()),
                            jacobian_ratio_below_point2=int((data[:,13]<.2).sum())),
                worst_shape_cells=[dict(zip(names,row.tolist())) for row in data[np.argsort(data[:,5])[:10]]],
                smallest_cells=[dict(zip(names,row.tolist())) for row in data[np.argsort(data[:,4])[:10]]],
                notes=['Corner quality q=4*sqrt(3)*A/sum(edge_length^2); equilateral=1.',
                       'Corner angle/shape metrics use vertex chords, not curved edge tangents.',
                       'Jacobian metrics sample the high-order map; positivity between samples is not certified.',
                       'Threshold counts are diagnostics, not universal acceptance criteria.'])
    (output/'quality.json').write_text(json.dumps(report,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fr-naca-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import matplotlib.colors as colors
    triangles=np.concatenate([b.geometry[:,:3,:].numpy() for b in mesh.blocks])
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for ax,values,title in [(axes[0,0],data[:,5],'Corner shape quality: equilateral=1'),
                            (axes[0,1],data[:,6],'Minimum corner angle (deg)'),
                            (axes[1,0],data[:,13],'Curved Jacobian min/max ratio')]:
        artist=PolyCollection(triangles,array=values,cmap='viridis',linewidths=.08,edgecolors='#333333')
        ax.add_collection(artist);ax.set_xlim(-.15,1.2);ax.set_ylim(-.4,.4);ax.set_aspect('equal')
        ax.set_title(title);fig.colorbar(artist,ax=ax)
    axes[1,1].hist(data[:,6],bins=40);axes[1,1].set_xlabel('Minimum corner angle (deg)');axes[1,1].set_ylabel('Cell count')
    fig.tight_layout();fig.savefig(output/'quality.png',dpi=180);plt.close(fig)
    summary=['# 三角形网格质量评估','',f'网格：`{mesh_path}`；{len(data)} 个单元。','',
             '| 指标 | 最小值 | 中位数 | 最大值 |','|---|---:|---:|---:|']
    for name in ['shape_quality','min_angle_deg','aspect_normalized','jacobian_min_max_ratio','scaled_jacobian_min','adjacent_sqrt_area_ratio']:
        v=metrics[name];summary.append(f'| {name} | {v["min"]:.6g} | {v["median"]:.6g} | {v["max"]:.6g} |')
    summary+=['',f'非正采样 Jacobian：{report["counts"]["sampled_nonpositive_jacobian"]}；最小角 <10°：{report["counts"]["min_angle_below_10"]}；<20°：{report["counts"]["min_angle_below_20"]}。',
              '', '形状质量为 1 表示等边三角形；越接近零越扁。角度、长宽比基于顶点弦边；曲边几何另检查 Jacobian。',
              '所有阈值只是排查线索；高阶几何采样不能证明采样点之间处处不翻转。',
              '完整分布、最差单元坐标和最小单元见 quality.json、cells.csv；质量良好也不能替代离散稳定性和边界条件验证。']
    (output/'report.md').write_text('\n'.join(summary)+'\n')
    print(json.dumps({k:report[k] for k in ['cells','counts']}),flush=True)
    print(json.dumps({name:metrics[name] for name in ['shape_quality','min_angle_deg','aspect_normalized','jacobian_min_max_ratio']},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,default=Path(__file__).parent/'results/mesh/downstream40.msh')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results/mesh_quality_downstream40')
    a=p.parse_args();evaluate(a.mesh,a.output)
