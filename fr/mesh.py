"""Conforming 2D mixed triangle/quad meshes; ASCII Gmsh 2.2 and 4.1 reader."""
from dataclasses import dataclass
from pathlib import Path
import shlex
import torch
from .reference import Basis, gmsh_reference_nodes

# Complete Lagrange elements; incomplete/serendipity elements are rejected.
ELEMENTS={1:('line',1,2),8:('line',2,3),26:('line',3,4),27:('line',4,5),28:('line',5,6),
          2:('tri',1,3),9:('tri',2,6),21:('tri',3,10),23:('tri',4,15),25:('tri',5,21),
          3:('quad',1,4),10:('quad',2,9),36:('quad',3,16),37:('quad',4,25),38:('quad',5,36),15:('point',0,1)}


@dataclass
class CellBlock:
    kind: str
    order: int
    vertices: torch.Tensor
    geometry: torch.Tensor
    regions: torch.Tensor


class Mesh2D:
    def __init__(self,points,blocks,edge_tags=None,physical_names=None):
        self.points=torch.as_tensor(points,dtype=torch.float64)
        self.blocks=blocks
        self.edge_tags={} if edge_tags is None else dict(edge_tags)
        self.physical_names={} if physical_names is None else dict(physical_names)
        if self.points.ndim!=2 or self.points.shape[1]!=2 or not torch.isfinite(self.points).all():
            raise ValueError('finite planar (nodes,2) points required')
        if not blocks: raise ValueError('mesh contains no triangle or quad elements')
        self.nelements=sum(len(b.vertices) for b in blocks)
        self._orient()
        self._connect()

    def _orient(self):
        for b in self.blocks:
            g=Basis(b.kind,b.order,nodes=gmsh_reference_nodes(b.kind,b.order))
            center=torch.tensor([[1/3,1/3] if b.kind=='tri' else [.5,.5]],dtype=torch.float64)
            xr=torch.einsum('qi,eik->eqk',g.evaluate(center,0),b.geometry)[:,0]
            xs=torch.einsum('qi,eik->eqk',g.evaluate(center,1),b.geometry)[:,0]
            determinant=xr[:,0]*xs[:,1]-xr[:,1]*xs[:,0]
            if (determinant.abs()<1e-14).any(): raise ValueError('degenerate element')
            reverse=determinant<0
            if reverse.any():
                nodes=g.nodes
                permutation=[torch.linalg.vector_norm(nodes-point.flip(0),dim=-1).argmin().item() for point in nodes]
                b.geometry[reverse]=b.geometry[reverse][:,permutation]
                permutation=[0,2,1] if b.kind=='tri' else [0,3,2,1]
                b.vertices[reverse]=b.vertices[reverse][:,permutation]

    def _connect(self):
        owners={}
        self.face_records=[]
        self.face_ids=[]
        for bi,b in enumerate(self.blocks):
            nf=3 if b.kind=='tri' else 4
            ids=[]
            for e,vertices in enumerate(b.vertices.tolist()):
                row=[]
                for f in range(nf):
                    edge=(vertices[f],vertices[(f+1)%nf])
                    index=len(self.face_records)
                    self.face_records.append((bi,e,f,edge))
                    owners.setdefault(tuple(sorted(edge)),[]).append(index)
                    row.append(index)
                ids.append(row)
            self.face_ids.append(torch.tensor(ids,dtype=torch.long))
        n=len(self.face_records)
        self.neighbor=torch.arange(n)
        self.flip=torch.zeros(n,dtype=torch.bool)
        self.boundary=torch.ones(n,dtype=torch.bool)
        self.face_tags=torch.zeros(n,dtype=torch.long)
        for edge,indices in owners.items():
            if len(indices)>2: raise ValueError('nonmanifold edge has more than two owners')
            for i in indices: self.face_tags[i]=self.edge_tags.get(edge,0)
            if len(indices)==2:
                i,j=indices
                self.neighbor[i],self.neighbor[j]=j,i
                a,b=self.face_records[i][-1],self.face_records[j][-1]
                self.flip[i]=self.flip[j]=(a==b[::-1])
                self.boundary[i]=self.boundary[j]=False
        self.nfaces=n

    @classmethod
    def from_cells(cls,points,cells,regions=None,edge_tags=None,physical_names=None):
        points=torch.as_tensor(points,dtype=torch.float64)
        regions=[1]*len(cells) if regions is None else regions
        grouped={}
        for cell,region in zip(cells,regions,strict=True):
            if len(cell) not in (3,4): raise ValueError('cells must have three or four vertices')
            kind='tri' if len(cell)==3 else 'quad'
            grouped.setdefault(kind,[]).append((list(cell),int(region)))
        blocks=[]
        for kind,rows in grouped.items():
            vertices=torch.tensor([c for c,r in rows],dtype=torch.long)
            blocks.append(CellBlock(kind,1,vertices,points[vertices].clone(),torch.tensor([r for c,r in rows])))
        return cls(points,blocks,edge_tags,physical_names)

    @classmethod
    def rectangle(cls,nx,ny=None,*,kind='quad',jitter=0.,seed=0,bounds=((0.,1.),(0.,1.))):
        ny=nx if ny is None else ny
        if nx<1 or ny<1 or kind not in ('tri','quad','mixed') or not 0<=jitter<.45:
            raise ValueError('positive cell counts, kind tri/quad/mixed, jitter in [0,.45) required')
        generator=torch.Generator().manual_seed(seed)
        points=[]
        for j in range(ny+1):
            for i in range(nx+1):
                offsets=(torch.rand(2,generator=generator,dtype=torch.float64)*2-1)*jitter if 0<i<nx and 0<j<ny else torch.zeros(2)
                points.append([bounds[0][0]+(i+float(offsets[0]))/nx*(bounds[0][1]-bounds[0][0]),
                               bounds[1][0]+(j+float(offsets[1]))/ny*(bounds[1][1]-bounds[1][0])])
        cells=[]
        edge_tags={}
        for j in range(ny):
            for i in range(nx):
                a=j*(nx+1)+i
                quad=[a,a+1,a+nx+2,a+nx+1]
                if kind=='tri' or kind=='mixed' and (i+j)%2:
                    cells.extend(([quad[0],quad[1],quad[2]],[quad[0],quad[2],quad[3]]))
                else: cells.append(quad)
        for i in range(nx):
            edge_tags[(i,i+1)]=1
            a=ny*(nx+1)+i
            edge_tags[(a,a+1)]=3
        for j in range(ny):
            a=j*(nx+1)
            edge_tags[(a,a+nx+1)]=4
            a+=nx
            edge_tags[(a,a+nx+1)]=2
        return cls.from_cells(points,cells,edge_tags=edge_tags,
                              physical_names={(1,1):'bottom',(1,2):'right',(1,3):'top',(1,4):'left'})

    def mapped(self,function,order=3):
        """Elevate geometry and apply a shared smooth mapping; preserve topology/tags."""
        blocks=[]
        for b in self.blocks:
            basis=Basis(b.kind,b.order,nodes=gmsh_reference_nodes(b.kind,b.order))
            target=gmsh_reference_nodes(b.kind,order)
            xy=torch.einsum('qi,eik->eqk',basis.evaluate(target),b.geometry)
            x,y=function(xy[...,0],xy[...,1])
            blocks.append(CellBlock(b.kind,order,b.vertices.clone(),torch.stack((x,y),-1),b.regions.clone()))
        x,y=function(self.points[:,0],self.points[:,1])
        return Mesh2D(torch.stack((x,y),-1),blocks,self.edge_tags,self.physical_names)

    @classmethod
    def read(cls,path):
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
        if any(abs(z)>1e-12 for x,y,z in nodes.values()): raise ValueError('mesh must lie in the XY plane')
        ids={tag:i for i,tag in enumerate(nodes)}
        points=torch.tensor([(x,y) for x,y,z in nodes.values()],dtype=torch.float64)
        grouped={}
        edges={}
        for typ,tags,physical in elements:
            if typ not in ELEMENTS: raise ValueError(f'unsupported Gmsh element type {typ}; use complete 2D Lagrange triangles/quads')
            kind,order,count=ELEMENTS[typ]
            if len(tags)!=count: raise ValueError('incorrect element node count')
            indices=[ids[t] for t in tags]
            if kind=='line': edges[tuple(sorted(indices[:2]))]=physical
            elif kind in ('tri','quad'):
                grouped.setdefault((kind,order),[]).append((indices,physical))
        blocks=[]
        for (kind,order),rows in grouped.items():
            nv=3 if kind=='tri' else 4
            indices=torch.tensor([c for c,r in rows])
            blocks.append(CellBlock(kind,order,indices[:,:nv].clone(),points[indices],torch.tensor([r for c,r in rows])))
        return cls(points,blocks,edges,names)
