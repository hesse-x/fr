"""Gmsh command-line integration; no Python gmsh/numpy dependency."""
from pathlib import Path
import shlex
import shutil
import subprocess


def generate_mesh(geometry,output,*,order=3,version='4.1',dimension=2):
    executable=shutil.which('gmsh')
    if executable is None: raise RuntimeError('gmsh executable not found in PATH')
    if version not in ('2.2','4.1'): raise ValueError('MSH version must be 2.2 or 4.1')
    if dimension not in (2,3): raise ValueError('mesh dimension must be 2 or 3')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([executable,str(geometry),f'-{dimension}','-order',str(order),'-format',
                           'msh2' if version=='2.2' else 'msh4','-setnumber','Mesh.Binary','0',
                           '-o',str(output),'-v','2'],capture_output=True,text=True)
    if result.returncode or not output.exists():
        raise RuntimeError('Gmsh failed: '+result.stdout+result.stderr)
    return output


def read_ascii_mesh(path):
    """Read nodes, physical names and elements from an ASCII MSH file."""
    raw=Path(path).read_bytes()
    if b'$MeshFormat' not in raw: raise ValueError('missing Gmsh MeshFormat section')
    header_lines=raw.split(b'$MeshFormat',1)[-1].splitlines()
    if len(header_lines)<2: raise ValueError('incomplete Gmsh MeshFormat section')
    header=header_lines[1].split()
    if len(header)<2 or header[1]!=b'0': raise ValueError('only ASCII MSH files are supported; export with Mesh.Binary=0')
    text=raw.decode('utf-8')
    sections={}
    lines=iter(text.splitlines())
    for line in lines:
        if line.startswith('$') and not line.startswith('$End'):
            name=line[1:]
            content=[]
            for value in lines:
                if value==f'$End{name}': break
                content.append(value)
            sections[name]=content
    version=sections['MeshFormat'][0].split()[0]
    names={}
    for line in sections.get('PhysicalNames',[])[1:]:
        dim,tag,name=shlex.split(line)
        names[(int(dim),int(tag))]=name
    nodes={}
    elements=[]
    if version=='2.2':
        for line in sections['Nodes'][1:]:
            tag,x,y,z=line.split()
            nodes[int(tag)]=(float(x),float(y),float(z))
        for line in sections['Elements'][1:]:
            values=list(map(int,line.split()))
            _,typ,ntags=values[:3]
            physical=values[3] if ntags else 0
            elements.append((typ,values[3+ntags:],physical))
    elif version=='4.1':
        entity_physical={}
        entity_lines=iter(sections.get('Entities',[]))
        counts=list(map(int,next(entity_lines,'0 0 0 0').split()))
        for dim,count in enumerate(counts):
            for _ in range(count):
                values=next(entity_lines).split()
                offset=4 if dim==0 else 7
                nphysical=int(values[offset])
                if nphysical>1: raise ValueError('multiple physical groups on one entity are ambiguous; use one tag per entity')
                entity_physical[(dim,int(values[0]))]=int(values[offset+1]) if nphysical else 0
        tokens=iter(' '.join(sections['Nodes']).split())
        nblocks,total,minimum,maximum=[int(next(tokens)) for _ in range(4)]
        for _ in range(nblocks):
            dim,entity,parametric,count=[int(next(tokens)) for _ in range(4)]
            tags=[int(next(tokens)) for _ in range(count)]
            for tag in tags:
                xyz=tuple(float(next(tokens)) for _ in range(3))
                for _ in range(dim if parametric else 0): next(tokens)
                nodes[tag]=xyz
        iterator=iter(sections['Elements'])
        nblocks,total,minimum,maximum=map(int,next(iterator).split())
        for _ in range(nblocks):
            dim,entity,typ,count=map(int,next(iterator).split())
            for _ in range(count):
                values=list(map(int,next(iterator).split()))
                elements.append((typ,values[1:],entity_physical.get((dim,entity),0)))
    else: raise ValueError('supported MSH versions are ASCII 2.2 and 4.1')
    return nodes, elements, names
