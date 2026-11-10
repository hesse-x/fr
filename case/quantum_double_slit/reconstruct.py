"""Connect the full image trace across printed dash gaps, without peak anchors."""
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator


def complete_trace(reference):
    raw=np.loadtxt(Path(reference),delimiter=",",skiprows=1)
    # Only source-image paths and the explicitly flagged normalization point.
    # Tails merge into the printed horizontal axis; their display value zero
    # is a baseline-limited value, not a recovered author measurement.
    samples=np.vstack(([-80.,0.],raw[:,:2],[80.,0.]))
    samples=samples[np.argsort(samples[:,0])]
    angle=np.linspace(-80,80,3201)
    curve=np.maximum(0,PchipInterpolator(samples[:,0],samples[:,1])(angle))
    nearest=np.abs(angle[:,None]-raw[None,:,0]).argmin(axis=1)
    distance=abs(angle-raw[nearest,0])
    status=np.where(distance<=.075,raw[nearest,4].astype(int),4)
    status[(angle<raw[0,0])|(angle>raw[-1,0])]=5
    return angle,curve,status


def reconstruct(reference):
    angle,curve,_=complete_trace(reference)
    return angle,curve,np.empty((0,2))
