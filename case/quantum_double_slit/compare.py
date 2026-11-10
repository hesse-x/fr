"""Overlay FR results with pixels digitized from the paper's Fig. 2 dashed curve."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .reconstruct import complete_trace
from .digitize import xy as data_to_pixel


def compare(runs, output, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reference=Path(reference)
    paper=np.loadtxt(reference,delimiter=',',skiprows=1)
    completed_angles,completed,completed_status=complete_trace(reference)
    observed=np.isin(paper[:,4],[1,2])
    measured=paper[observed]
    rows=[];curves=[]
    fig,ax=plt.subplots(figsize=(11,5))
    for run in map(Path,runs):
        report=json.loads((run/'report.json').read_text())
        data=np.load(run/'solution.npz')
        angles,profile=data['angles'],data['profiles'][-1]
        parameters=report['parameters'];last=report['history'][-1]
        label=f"FR p={parameters['p']}, h={parameters['h']:g}"
        ax.plot(angles,profile,label=label,lw=1.4)
        curves.append(profile)
        difference=np.interp(measured[:,0],angles,profile)-measured[:,1]
        # Readout sensitivity, not a statistical confidence interval: let each
        # traced ink location vary over its measured horizontal span and 3 pixels
        # vertically. Overlapping ink cannot identify a unique numerical curve.
        delta=.15+measured[:,5]/4.2
        samples=measured[:,0,None]+delta[:,None]*np.linspace(-1,1,31)
        prediction=np.interp(samples,angles,profile)
        lower=measured[:,1]-.006;upper=measured[:,1]+.006
        best=np.maximum(np.maximum(prediction.min(1)-upper,lower-prediction.max(1)),0)
        worst=np.maximum(abs(prediction.min(1)-upper),abs(prediction.max(1)-lower))
        rows.append(dict(run=str(run),label=label,transmitted_probability=last['transmitted_probability'],
                         cells=[len(data['x'])//(parameters['p']+1),len(data['y'])//(parameters['p']+1)],
                         complex_dofs=len(data['x'])*len(data['y']),wall_seconds=report['wall_seconds'],
                         probability_change=last['probability']-1,initial_energy=report['initial_energy'],
                         energy_change=last['energy']-report['initial_energy'],
                         late_profile_relative_l2=report['late_profile_relative_l2'],
                         reference_version=2,digitized_point_count=len(measured),
                         traced_point_rmse=float(np.sqrt(np.mean(difference**2))),
                         readout_sensitivity_rmse_range=[float(np.sqrt(np.mean(best**2))),float(np.sqrt(np.mean(worst**2)))],
                         precise_method_ranking_supported=False,
                         positive_peaks=last['positive_peaks']))
    ax.plot(completed_angles,completed,'k--',lw=1.3,label='Paper Fig. 2: full image trace (overlap flagged)')
    missing=np.isin(completed_status,[3,4,5])
    ax.plot(completed_angles,np.where(missing,completed,np.nan),color='gray',ls=':',lw=1.6,label='Dash gaps / occlusion / baseline')
    ax.set(xlabel='Angle (degrees)',ylabel='P(theta) / P(0)',xlim=(-80,80),ylim=(0,1.15))
    ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(output/'comparison.png',dpi=200);plt.close(fig)
    if len(curves)>1:
        fig,ax=plt.subplots(figsize=(11,4))
        for row,curve in zip(rows[:-1],curves[:-1]):
            ax.plot(angles,curve-curves[-1],label=row['label'])
        ax.axhline(0,color='gray',lw=.6)
        ax.set(xlabel='Angle (degrees)',ylabel='Intensity difference from final supplied run',xlim=(-80,80))
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend();ax.grid(alpha=.15)
        fig.tight_layout();fig.savefig(output/'grid_difference.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,7))
    ax.imshow(plt.imread(reference.with_suffix('.png')))
    ax.scatter(paper[:,2],paper[:,3],s=10,facecolors='none',edgecolors='red',linewidths=.7)
    pixel_x,pixel_y=data_to_pixel(completed_angles,completed)
    ax.plot(pixel_x,pixel_y,color='cyan',lw=.8,alpha=.8)
    ax.axis('off');ax.set_title('Version 2 image trace; overlap and gaps are not exact author data')
    fig.tight_layout();fig.savefig(output/'digitization_check.png',dpi=180);plt.close(fig)
    np.savetxt(output/'comparison.csv',np.column_stack((angles,*curves)),delimiter=',',
               header='angle_degrees,'+','.join(Path(run).name for run in runs),comments='')
    point_values=[np.interp(paper[:,0],angles,curve) for curve in curves]
    np.savetxt(output/'paper_reconstructed.csv',np.column_stack((completed_angles,completed,completed_status)),delimiter=',',
               header='angle_degrees,traced_intensity,status',comments='')
    np.savetxt(output/'paper_point_comparison.csv',np.column_stack((paper[:,:2],paper[:,4],*point_values)),delimiter=',',
               header='angle_degrees,paper_image_trace,status,'+','.join(Path(run).name for run in runs),comments='')
    for row,curve in zip(rows,curves):
        row['profile_relative_l2_vs_last_run']=float(np.linalg.norm(curve-curves[-1])/np.linalg.norm(curves[-1]))
    (output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    text=['# 量子双缝波包散射与论文图 2 对照','',
          '来源：[De Raedt & Michielsen, Computers in Physics 8, 600–607](https://doi.org/10.1063/1.168483)。',
          f'图像轨迹共 {len(paper)} 个角度分箱，其中 {len(measured)} 个为可见墨迹（包含无法分离的虚实线重叠），其余为虚线间隙或坐标轴遮挡。',
          '| 计算 | 第一侧峰角度 | 第一侧峰高度 | 图像轨迹 RMSE（描述值） | 读图敏感性范围 |',
          '|---|---:|---:|---:|---:|']
    for row in rows:
        peak=next(p for p in row['positive_peaks'] if p['angle']>1)
        lo,hi=row['readout_sensitivity_rmse_range']
        text.append(f"| {row['label']} | {peak['angle']:.1f}° | {peak['height']:.4f} | {row['traced_point_rmse']:.4f} | {lo:.4f}–{hi:.4f} |")
    text+=['','读图敏感性范围不是统计置信区间。它让每个读数在可见线宽和坐标标定误差内变化；不应据此作精细的数值方法排名。']
    text+=['','| 网格 | 单元数 | 复自由度 | 右侧区域概率 | 与末行角分布相对 L2 差 |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['label']} | {row['cells'][0]}×{row['cells'][1]} | {row['complex_dofs']:,} | {row['transmitted_probability']:.8f} | {row['profile_relative_l2_vs_last_run']:.3%} |")
    text+=['','末行仅用于网格间数值对照，不视为精确解。']
    if len(curves)>1:
        text+=['','![网格间角分布差异](grid_difference.png)']
    text+=['','![FR 与论文全曲线叠加](comparison.png)','',
           '黑色虚线来自整条图像轨迹；灰色点线为虚线空隙、坐标轴遮挡和与基线合并的尾部。',
           '没有设置估计峰位、峰高或谷值锚点，也没有用 FR 解或衍射公式补线。',
           '可见实线与虚线共用墨迹的区域仍无法唯一分离，轨迹的线宽范围反映了这种不确定性。',
           '评分仅使用可见墨迹行，排除虚线间隙、中心归一化点和尾部显示延拓。',
           '`paper_reconstructed.csv` 导出整条轨迹及状态码；所有源像素坐标在 `reference/figure2.csv`。','',
           '![逐段轨迹回投原图](digitization_check.png)','',
           '更清楚的分类回投图和可复现脚本说明见仓库 `reference/trace_overlay.png` 与 `reference/README.md`。']
    (output/'summary.md').write_text('\n'.join(text)+'\n')
    default_output=Path(__file__).resolve().parents[2]/'results'/'quantum_double_slit'
    if output.resolve()==default_output:
        shutil.copyfile(output/'comparison.png',Path(__file__).with_name('comparison.png'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',default=['results/quantum_double_slit/fr_p4_h05'])
    parser.add_argument('--reference',default=str(Path(__file__).with_name('reference')/'figure2.csv'))
    parser.add_argument('--output',default='results/quantum_double_slit')
    args=parser.parse_args();compare(args.runs,args.output,args.reference)


if __name__=='__main__':main()
