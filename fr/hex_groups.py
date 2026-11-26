"""Spatial tiles, physical cell ordering and pruned FR/LDG dependency plans."""
from copy import copy
import numpy as np


def pack_trace_permutations(neighbor_points):
    """Four uint32 descriptors per face: four 4-bit offsets plus a load kind.

    Kind 0 is contiguous, 1 is reverse-contiguous, 2 is a four-lane gather.
    Descriptors only encode local face-point order, never global cell indices.
    """
    neighbor_points=np.asarray(neighbor_points)
    if neighbor_points.ndim!=3 or neighbor_points.shape[1:]!=(6,16) or not np.issubdtype(neighbor_points.dtype,np.integer):
        raise ValueError('integer (cells,6,16) point map required')
    result=np.empty(neighbor_points.shape[:2]+(4,),dtype=np.uint32)
    lane=np.arange(4,dtype=np.int16);shift=np.arange(4,dtype=np.uint32)*4
    for start in range(0,len(result),1024):
        values=neighbor_points[start:start+1024]
        if np.any(values<0):raise ValueError('nonnegative point indices required')
        relative=(values%16).astype(np.uint8)
        if not np.all(np.sort(relative,axis=-1)==np.arange(16,dtype=np.uint8)):
            raise ValueError('each face must be a permutation of sixteen points')
        relative=relative.reshape(relative.shape[:2]+(4,4))
        forward=np.all(relative==relative[...,:1].astype(np.int16)+lane,axis=-1)
        reverse=np.all(relative==relative[...,:1].astype(np.int16)-lane,axis=-1)
        kind=np.where(forward,0,np.where(reverse,1,2)).astype(np.uint32)
        result[start:start+1024]=np.bitwise_or.reduce(relative.astype(np.uint32)<<shift,axis=-1)|(kind<<16)
    return result


def cartesian_cell_indices(mesh,lines,selected):
    """Recover global Cartesian indices, certifying selected cell extents."""
    selected=np.asarray(selected,dtype=bool)
    cells=mesh.cells()
    if selected.shape!=(len(cells),) or not selected.any() or len(lines)!=3:
        raise ValueError('nonempty cell selection and three Cartesian edge arrays required')
    xyz=mesh.nodes[cells[selected]]
    result=np.full((len(cells),3),-1,dtype=np.int64)
    for axis,edges in enumerate(lines):
        edges=np.asarray(edges,dtype=float)
        if edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0):
            raise ValueError('Cartesian edges must increase strictly')
        lower=xyz[:,:,axis].min(axis=1);upper=xyz[:,:,axis].max(axis=1)
        index=np.searchsorted(edges,(lower+upper)/2,side='right')-1
        if np.any(index<0) or np.any(index>=len(edges)-1):raise ValueError('cell outside Cartesian lattice')
        tol=1e-11*max(1.,float(np.abs(edges).max()))
        if not (np.allclose(lower,edges[index],rtol=0,atol=tol) and np.allclose(upper,edges[index+1],rtol=0,atol=tol)):
            raise ValueError('selected cells do not match Cartesian lattice')
        on_lower=np.abs(xyz[:,:,axis]-edges[index,None])<=tol
        on_upper=np.abs(xyz[:,:,axis]-edges[index+1,None])<=tol
        if not np.all(on_lower|on_upper):raise ValueError('selected cells are not Cartesian hexahedra')
        result[selected,axis]=index
    return result


class HexGroups:
    """One owner per cell; halo work never changes the numerical face rule.

    Neighbor halos supply the conserved and primal traces. Groups compute
    gradients only for owned cells and publish the required dual viscous traces.
    Optional spatial_mask selects cells grouped into physical bins of the given
    spatial_width, across block boundaries; other cells retain logical tiles.
    index_coordinates uses global integer lattice indices instead of physical
    widths. Owned face pairs support immediate two-sided local lifts.
    """
    def __init__(self,geometry,tile=(4,4,4),*,reorder=True,spatial_mask=None,spatial_width=None,index_coordinates=None):
        if len(tile)!=3 or any(not isinstance(n,int) or n<1 for n in tile):
            raise ValueError('three positive integer tile dimensions required')
        original=geometry
        mask=None
        if index_coordinates is not None:
            indices=np.asarray(index_coordinates)
            if spatial_mask is not None or spatial_width is not None:raise ValueError('choose physical bins or indexed groups')
            if indices.shape!=(original.ne,3) or not np.issubdtype(indices.dtype,np.integer):raise ValueError('integer Cartesian indices required')
            mask=np.all(indices>=0,axis=1)
            if not mask.any() or np.any(~mask&np.any(indices!=-1,axis=1)):raise ValueError('unselected indices must all be -1')
            if len(np.unique(indices[mask],axis=0))!=int(mask.sum()):raise ValueError('duplicate Cartesian cell indices')
        elif spatial_mask is not None:
            mask=np.asarray(spatial_mask,dtype=bool)
            width=np.asarray(spatial_width,dtype=float)
            if mask.shape!=(original.ne,) or not mask.any() or width.shape!=(3,) or not np.isfinite(width).all() or (width<=0).any():
                raise ValueError('spatial grouping requires a nonempty cell mask and three positive finite widths')
        elif spatial_width is not None:
            raise ValueError('spatial width requires a cell mask')
        owners=[];offset=0
        for block in original.mesh.blocks:
            shape=tuple(np.array(block.shape)-1)
            ids=np.arange(offset,offset+np.prod(shape)).reshape(shape)
            for i in range(0,shape[0],tile[0]):
                for j in range(0,shape[1],tile[1]):
                    for k in range(0,shape[2],tile[2]):
                        selected=ids[i:i+tile[0],j:j+tile[1],k:k+tile[2]].ravel()
                        if mask is not None:selected=selected[~mask[selected]]
                        if len(selected):owners.append(selected)
            offset+=int(np.prod(shape))
        if offset!=original.ne:raise ValueError('logical blocks do not match geometry cell order')
        if mask is not None:
            selected=np.flatnonzero(mask)
            # Shared physical bins cross logical-block boundaries. They change
            # execution ownership only; all face rules still use mesh neighbors.
            if index_coordinates is not None:bins=indices[selected]//np.asarray(tile)
            else:
                xyz=original.mesh.nodes[original.cells[selected]].mean(axis=1)
                bins=np.floor((xyz-xyz.min(axis=0))/width+1e-9).astype(np.int64)
            order=np.lexsort((selected,bins[:,2],bins[:,1],bins[:,0]))
            cuts=np.flatnonzero(np.any(np.diff(bins[order],axis=0),axis=1))+1
            spatial_owners=np.split(selected[order],cuts)
            owners=spatial_owners+owners
        self.order=np.concatenate(owners) if reorder else np.arange(original.ne)
        self.inverse=np.argsort(self.order)
        if not np.array_equal(np.sort(self.order),np.arange(original.ne)):
            raise ValueError('tile ordering must own every cell exactly once')
        s=copy(original)
        for name in ('cells','J','C','S','tags','primal_inside','volume'):
            setattr(s,name,np.ascontiguousarray(getattr(original,name)[self.order]))
        fn=original.face_neighbor[self.order]
        s.face_neighbor=np.ascontiguousarray(self.inverse[fn//6]*6+fn%6)
        nb=original.neighbor_points[self.order]
        s.neighbor_points=np.ascontiguousarray(self.inverse[nb//96]*96+nb%96)
        s.boundary_faces=np.flatnonzero(s.tags.ravel()>0)
        s.wall_faces=np.flatnonzero(s.tags.ravel()==1)
        s.boundary_index=np.full(s.ne*6,-1,dtype=np.int64)
        s.boundary_index[s.boundary_faces]=np.arange(len(s.boundary_faces))
        s.boundary_index=s.boundary_index.reshape(s.ne,6)
        self.geometry=s;self.original_geometry=original;self.tile=tuple(tile)
        # Flattened per-group records use offsets, avoiding Python in hot loops.
        pids=[];uslots=[];wslots=[];vslots=[];wnb=[];unb=[];vsrc=[];records=[]
        max_scratch=primitive_total=0
        shared_sources=[];shared_peers=[];owned_offsets=[0];internal_faces=0
        masks=s.primal_inside.astype(bool);tags=s.tags;neighbor=s.face_neighbor
        def slots(cells,selected):
            lookup={int(face):i for i,face in enumerate(selected)}
            arr=np.full((len(cells),6),-1,dtype=np.int64)
            for i,e in enumerate(cells):
                for f in range(6):arr[i,f]=lookup.get(int(e*6+f),-1)
            return arr,lookup
        for old_owner in owners:
            no=len(old_owner);owned=self.inverse[old_owner]
            own_faces=owned[:,None]*6+np.arange(6)
            other=neighbor[owned]
            interior=tags[owned]==0
            needs_neighbor=interior&masks[owned]
            needs_primal=interior&~masks[owned]
            primitive=np.r_[owned,np.setdiff1d(other[interior]//6,owned)]
            np_=len(primitive)
            selected_u=np.unique(np.r_[own_faces.ravel(),other[interior]])
            selected_w=np.unique(np.r_[own_faces[~needs_neighbor],other[needs_primal]])
            local_v=~needs_neighbor|np.isin(other//6,owned)
            sources=np.where(needs_neighbor,other,own_faces)
            selected_v=np.unique(sources[local_v])
            u,ulookup=slots(primitive,selected_u)
            w,wlookup=slots(primitive,selected_w)
            v,vlookup=slots(owned,selected_v)
            wn=np.full((no,6),-1,dtype=np.int64)
            un=np.full((no,6),-1,dtype=np.int64);vs=np.full((no,6),-1,dtype=np.int64)
            for i,e in enumerate(owned):
                for f in range(6):
                    if needs_primal[i,f]:wn[i,f]=wlookup[int(other[i,f])]
                    if interior[i,f]:un[i,f]=ulookup[int(other[i,f])]
                    if local_v[i,f]:vs[i,f]=vlookup[int(sources[i,f])]
            nu,nw,nv=len(selected_u),len(selected_w),len(selected_v)
            pair_sources=np.zeros((no,6),dtype=np.int64);pair_peers=np.full((no,6),-1,dtype=np.int64);pairs={}
            positions={int(e):i for i,e in enumerate(owned)}
            for i,e in enumerate(owned):
                for f in range(6):
                    if tags[e,f]:continue
                    peer=int(neighbor[e,f]);j=positions.get(peer//6)
                    if j is None:continue
                    key=tuple(sorted((int(e*6+f),peer)))
                    slot=pairs.setdefault(key,len(pairs))+1
                    pair_sources[i,f]=slot if (i,f)<(j,peer%6) else -slot
                    pair_peers[i,f]=j*6+peer%6
            shared_sources.extend(pair_sources.ravel().tolist());owned_offsets.append(owned_offsets[-1]+no)
            shared_peers.extend(pair_peers.ravel().tolist())
            internal_faces+=len(pairs)
            # W and U nodes for owned cells, compact face
            # traces, and owned residuals. Per-element scratch is on the stack.
            scratch=no*14*64+nu*5*16+nw*4*16+nv*6*16
            records.append([no,np_,nu,nw,nv,len(pids),len(uslots),len(wslots),len(vslots),len(wnb),len(unb),len(vsrc),scratch])
            pids.extend(primitive.tolist());uslots.extend(u.ravel().tolist());wslots.extend(w.ravel().tolist())
            vslots.extend(v.ravel().tolist());wnb.extend(wn.ravel().tolist());unb.extend(un.ravel().tolist());vsrc.extend(vs.ravel().tolist())
            max_scratch=max(max_scratch,scratch);primitive_total+=np_
        self.records=np.asarray(records,dtype=np.int64)
        self.arrays=[self.records]+[np.asarray(v,dtype=np.int64) for v in (pids,uslots,wslots,vslots,wnb,unb,vsrc)]
        self.max_scratch_doubles=int(max_scratch)
        self.shared_sources=np.asarray(shared_sources,dtype=np.int64)
        self.shared_peers=np.asarray(shared_peers,dtype=np.int64)
        self.owned_offsets=np.asarray(owned_offsets,dtype=np.int64)
        self.metadata=dict(tile=list(tile),groups=len(records),owned_cells=original.ne,
            primitive_cells_with_repetition=primitive_total,primitive_work_ratio=primitive_total/original.ne,
            scratch_bytes_per_worker=int(max_scratch*8),
            scratch_bytes_percentiles=np.percentile(self.records[:,-1]*8,[0,50,95,100]).tolist(),
            plan_bytes=sum(a.nbytes for a in self.arrays)+self.shared_sources.nbytes+self.shared_peers.nbytes+self.owned_offsets.nbytes,reordered=bool(reorder),
            ordering='logical spatial tiles; owned cells contiguous' if reorder else 'logical spatial tiles; original storage',
            internal_group_faces=internal_faces,external_group_face_sides=original.ne*6-2*internal_faces,
            numerical_flux_evaluations_per_rhs=original.ne*6-internal_faces,
            shared_numerical_flux_computed_once=True,shared_numerical_flux_scope='group interiors only')
        if mask is not None:
            ends=np.cumsum([int(np.prod(np.array(b.shape)-1)) for b in original.mesh.blocks])
            crossing=sum(len(np.unique(np.searchsorted(ends,o,side='right')))>1 for o in spatial_owners)
            self.metadata.update(spatial_cells=int(mask.sum()),
                spatial_groups=len(spatial_owners),spatial_cross_block_groups=int(crossing),
                ordering=('global index tiles' if index_coordinates is not None else 'physical spatial bins')+' for selected cells; logical tiles for remainder; '+('group-owned storage' if reorder else 'original storage'))
            if index_coordinates is not None:self.metadata['index_tile']=list(tile)
            else:self.metadata['spatial_width']=width.tolist()
