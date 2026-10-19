"""Gmsh command-line integration; no Python gmsh/numpy dependency."""
from pathlib import Path
import shutil
import subprocess


def generate_mesh(geometry,output,*,order=3,version='4.1'):
    executable=shutil.which('gmsh')
    if executable is None: raise RuntimeError('gmsh executable not found in PATH')
    if version not in ('2.2','4.1'): raise ValueError('MSH version must be 2.2 or 4.1')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([executable,str(geometry),'-2','-order',str(order),'-format',
                           'msh2' if version=='2.2' else 'msh4','-setnumber','Mesh.Binary','0',
                           '-o',str(output),'-v','2'],capture_output=True,text=True)
    if result.returncode or not output.exists():
        raise RuntimeError('Gmsh failed: '+result.stdout+result.stderr)
    return output
