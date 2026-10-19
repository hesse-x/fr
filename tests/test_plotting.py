"""Verify physical sampling, observer isolation and actual image artifacts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch
from fr import CartesianFR,Mesh2D,MeshFR2D,normal_central,integrate
from fr.plotting import FieldSampler,PlotRecorder


class ObserverTests(unittest.TestCase):
    def test_observer_does_not_change_integration(self):
        initial=torch.tensor([1.],dtype=torch.float64)
        rhs=lambda U,t:-U
        baseline,stats=integrate(rhs,initial,.2,dt=.07)
        times=[]
        observed,other=integrate(rhs,initial,.2,dt=.07,monitor=lambda U,t:times.append(t))
        torch.testing.assert_close(observed,baseline,rtol=0,atol=0)
        self.assertEqual(stats,other)
        self.assertEqual(times,[0.,.07,.14,.2])


@unittest.skipUnless(importlib.util.find_spec('numpy'),'plot sampling requires NumPy')
class SamplingTests(unittest.TestCase):
    def cartesian(self,dim):
        return CartesianFR((2,3,2)[:dim],((0.,1.),(-1.,2.),(.2,2.2))[:dim],p=3,
                           flux=lambda U,a:U,flux_pair=lambda m,p:(m+p)/2,periodic=True)

    def test_cartesian_polynomial_and_middle_slice(self):
        for dim in (1,2,3):
            solver=self.cartesian(dim)
            field=(2+solver.coords[0]**3)
            if dim>1: field=field+2*solver.coords[1]**2
            if dim>2: field=field+3*solver.coords[2]
            sampler=FieldSampler(solver)
            sampled=sampler.state(field[...,None])[:,0].numpy()
            if dim==1: expected=2+sampler.x**3
            else:
                centers=sampler.polygons.mean(1)
                expected=2+centers[:,0]**3+2*centers[:,1]**2
                if dim==3: expected+=3*1.2
            import numpy as np
            np.testing.assert_allclose(sampled,expected,atol=3e-13,rtol=0)

    def test_curved_mixed_geometry_and_complex_sampling(self):
        mesh=Mesh2D.rectangle(2,kind='mixed',jitter=.1).mapped(lambda x,y:(x+.03*x*y,y+.02*x*x),3)
        solver=MeshFR2D(mesh,flux=lambda U,a:U,numerical_flux=normal_central,boundary=lambda c,U,n,tag,t:U)
        sampler=FieldSampler(solver)
        x,y=solver.coords
        U=(1+x+2*y)[:,None].to(torch.complex128)*(1+2j)
        result=sampler.state(U)
        expected=[]
        for b,L in sampler.parts:
            # Geometry evaluated at the same reference sampling sites.
            xy=b['geometry'].evaluate(b['basis'].nodes)@b['X']
            expected.append((L@((1+xy[...,0]+2*xy[...,1])[...,None])).reshape(-1,1)*(1+2j))
        torch.testing.assert_close(result,torch.cat(expected),atol=1e-12,rtol=0)
        self.assertEqual(len(result),len(sampler.polygons))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') and importlib.util.find_spec('PIL'),'optional plotting dependencies missing')
    def test_png_gif_endpoints_and_fixed_limits(self):
        from PIL import Image
        solver=self.cartesian(2)
        recorder=PlotRecorder(solver,{'Wave':lambda U:U[...,0]},.2,frames=4,fps=8)
        base=solver.coords[0]+solver.coords[1]
        for t in (0.,.05,.1,.15,.2): recorder((base+t)[...,None],t)
        with tempfile.TemporaryDirectory() as tmp:
            stem=Path(tmp)/'wave'
            recorder.finish(stem,(base+.2)[...,None],.2)
            with Image.open(stem.with_suffix('.png')) as image: self.assertGreater(image.width,100)
            with Image.open(stem.with_suffix('.gif')) as image: self.assertEqual(image.n_frames,4)
            metadata=json.loads(stem.with_name('wave_plot.json').read_text())
            self.assertEqual(metadata['snapshot_times'],[0.,.1,.15,.2])
            self.assertEqual(len(metadata['color_limits']['Wave']),2)


if __name__=='__main__': unittest.main()
