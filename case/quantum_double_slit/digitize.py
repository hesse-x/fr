"""Trace the scanned Fig. 2 in image coordinates, independent of simulation data.

Run: python -m case.quantum_double_slit.digitize
The calibration and lobe windows apply only to the bundled 1117x798 crop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator

# Calibrate only from printed axes and ticks, never from FR or an aperture model.
ticks=np.arange(-80,81,20.)
bottom=np.array([265,356,443,529.5,613.5,700.5,789,875.5,962.])
top=np.array([270.5,359.5,447.5,533.5,617.5,704.5,793.5,880.5,966.5])
def intensity(x,y):return (691.5-.0089*(x-615)-y)/558.5
def angle(x,y):
 p=intensity(x,y); xx=bottom+(top-bottom)*p/1.095
 return np.interp(x,xx,ticks)
def xy(theta,p):
 x=np.interp(theta,ticks,bottom)+(np.interp(theta,ticks,top)-np.interp(theta,ticks,bottom))*p/1.095
 return x,691.5-.0089*(x-615)-558.5*p

def digitize(image, output):
    image,output=Path(image),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    im=np.array(Image.open(image).convert('L'))
    if im.shape!=(798,1117):
        raise ValueError('calibration requires the bundled 1117x798 source crop')
    height,width=im.shape
    black=im<120
    # Remove known coordinate-axis strokes; curve crossings remain marked occluded.
    mask=black.copy()
    for y in range(height):
     cx=613.6+(691.5-y)*4/600
     mask[y,max(0,round(cx-1.4)):round(cx+1.4)+1]=False
    mask[409:416,603:634]=False
    mask[129:138,602:634]=False
    for tx in bottom:
     by=691.5-.0089*(tx-615)
     mask[round(by-18):round(by+3),round(tx-3):round(tx+3)+1]=False
    for x in range(width):mask[max(0,round(691.5-.0089*(x-615)-3)):,x]=False
    # Windows delimit lobes in IMAGE coordinates. Their peak height is measured
    # from ink, not specified as a numerical intensity anchor.
    windows=[(350,427,394,560),(427,487,463,470),(487,541,519,300),(541,591,570,170),
             (591,640,617,132),(640,691,666,170),(691,746,718,300),(746,806,775,470),(806,887,840,560)]
    labels,nlabels=label(mask); component_sizes=np.bincount(labels.ravel())
    all_points=[];branch_points=[];peaks=[]
    for li,(left,right,guess,ystart) in enumerate(windows):
     if li==4:
      peakx,peaky=xy(0,1);peakx=float(peakx);peaky=float(peaky)
     else:
      rx=18 if li in (0,8) else 9
      yy,xx=np.where(mask[ystart:685,guess-rx:guess+rx+1]);yy=yy+ystart;xx=xx+guess-rx
      topy=int(yy.min());sel=yy<=topy+3
      peakx=float(np.median(xx[sel]));peaky=float(np.median(yy[sel]))
     peaks.append((peakx,peaky))
     split=peakx
     if li!=4:
      yy,xx=np.where(mask[ystart:685,guess-18:guess+19]);yy=yy+ystart;xx=xx+guess-18
      big=component_sizes[labels[yy,xx]]>100
      if np.any(big):
       topbig=yy[big].min(); split=float(np.median(xx[big&(yy<=topbig+3)]))
     for side,(lo,hi) in enumerate([(left,int(np.floor(split))),(int(np.ceil(split)),right)]):
      data=[]
      for y in range(int(np.ceil(peaky))+2,688):
       xs=np.flatnonzero(mask[y,lo:hi+1])+lo
       if not len(xs):continue
       groups=np.split(xs,np.where(np.diff(xs)>1)[0]+1)
       groups=[g for g in groups if len(g)>=2]
       if not groups:continue
       # A separated thick dash is preferred over the thin continuous model line.
       small=[g for g in groups if 3<=component_sizes[labels[y,int(g[len(g)//2])]]<100]
       widths=np.array([len(g) for g in groups]);g=max(small,key=len) if small else groups[int(np.argmax(widths))]
       x=float(g.mean())
       status=1 if small else 2
       spread=max(1.5,(xs.max()-xs.min())/2)
       data.append((x,float(y),status,spread))
      data=np.asarray(data)
      if len(data)<10:raise RuntimeError((li,side,'insufficient ink'))
      rawx=data[:,0].copy()
      # Small scan-pixel smoothing along each monotone branch, not across a peak.
      length=min(15,len(data)//2*2-1)
      smoothed=savgol_filter(rawx,length,2)
      # Keep the path inside the local measured ink envelope, expanded by 1 pixel.
      data[:,0]=np.clip(smoothed,rawx-data[:,3]-1,rawx+data[:,3]+1)
      vals=[]
      for j,(x,y,status,spread) in enumerate(data):
       vals.append((angle(x,y),max(0,intensity(x,y)),x,y,status,spread,li,side))
      values=np.asarray(vals)
      resolved=values[values[:,4]==1]
      # On branches with separated dashes, use their measured centre positions
      # to bridge between dash fragments rather than reverting to the solid line.
      if len(resolved)>=4:
       ry=np.unique(resolved[:,3]); rx=np.array([np.median(resolved[resolved[:,3]==v,2]) for v in ry])
       interp=PchipInterpolator(ry,rx,extrapolate=False)
       inside=(values[:,3]>=ry[0])&(values[:,3]<=ry[-1])
       values[inside,2]=interp(values[inside,3])
       for j in np.flatnonzero(inside):
        xx,yy=values[j,2:4];values[j,0]=angle(xx,yy);values[j,1]=max(0,intensity(xx,yy))
        if values[j,4]!=1:values[j,4]=4
      branch_points.append(values);all_points.extend(values)
     # Peak from raster. The central normalized point is explicitly occluded.
     all_points.append((0 if li==4 else angle(peakx,peaky),1 if li==4 else intensity(peakx,peaky),peakx,peaky,3 if li==4 else 2,2,li,2))
    pts=np.asarray(all_points);pts=pts[np.argsort(pts[:,0])]
    # Reduce repeated pixel rows in equal ANGLE bins to avoid slope-weighted sampling.
    bins=np.floor((pts[:,0]+80)/.1).astype(int)
    result=[]
    for b in np.unique(bins):
     group=pts[bins==b]; v=np.median(group,axis=0); 
     v[4]=1 if np.all(group[:,4]==1) else (2 if np.any(group[:,4]==2) else (3 if np.any(group[:,4]==3) else 4))
     result.append(v)
    result=np.array(result);result=result[np.argsort(result[:,0])]

    columns='angle_degrees,normalized_intensity,pixel_x,pixel_y,status,pixel_halfwidth,lobe,branch'
    np.savetxt(output/'figure2.csv',result,delimiter=',',header=columns,comments='',fmt='%.8f')
    np.savetxt(output/'figure2_trace_rows.csv',pts,delimiter=',',header=columns,comments='',fmt='%.8f')
    metadata=dict(version=2,source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        source_size=[width,height],source_pdf_page=7,render_dpi=300,crop=[93,93,1210,891],
        angle_ticks=ticks.tolist(),bottom_tick_x=bottom.tolist(),top_tick_x=top.tolist(),
        raw_trace_rows=len(pts),angle_bins=len(result),angle_bin_width=.1,
        status_codes={1:'separated_dash',2:'overlap',3:'axis_normalization',4:'dash_gap'},
        status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(result[:,4],return_counts=True))},
        notes=['No FR solution or theoretical diffraction formula is read.',
               'Lobe windows constrain image regions, not numerical peak heights.',
               'Overlapping ink does not identify the authors numerical curve uniquely.',
               'No precise method ranking is justified by these image traces.'])
    (output/'digitization.json').write_text(json.dumps(metadata,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(14,10))
    ax.imshow(im,cmap='gray',vmin=0,vmax=255)
    colors={1:'lime',2:'deepskyblue',3:'magenta',4:'orange'}
    labels_text={1:'Separated dash',2:'Overlapping ink',3:'Axis / normalization',4:'Between dashes'}
    for code in (2,4,1,3):
        selected=result[result[:,4]==code]
        if len(selected):ax.scatter(selected[:,2],selected[:,3],s=4,c=colors[code],label=labels_text[code],alpha=.8)
    ax.set_xlim(235,990);ax.set_ylim(700,110);ax.legend(loc='upper left',fontsize=9)
    ax.set_title('Full image trace: observed ink and unresolved regions')
    fig.tight_layout();fig.savefig(output/'trace_overlay.png',dpi=180);plt.close(fig)
    print(json.dumps(metadata,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).with_name('reference')
    parser.add_argument('--image',default=str(root/'figure2.png'))
    parser.add_argument('--output',default=str(root))
    args=parser.parse_args();digitize(args.image,args.output)


if __name__=='__main__': main()
