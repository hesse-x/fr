import csv
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch

from case.maxwell_cavity import CavityMode, MaxwellCavityCase, export
from fr import Face, Maxwell


torch.set_num_threads(1)


class MaxwellVerification(unittest.TestCase):
    def test_flux_spectrum_and_roe_characteristics(self):
        eq=Maxwell(epsilon=2.3,mu=.7)
        eye=torch.eye(6,dtype=torch.float64)
        weights=torch.tensor([eq.epsilon]*3+[eq.mu]*3,dtype=torch.float64)
        S=torch.diag(weights.sqrt())
        inverse=torch.diag(1/weights.sqrt())
        torch.manual_seed(31)
        left,right=torch.randn(9,6,dtype=torch.float64),torch.randn(9,6,dtype=torch.float64)
        for axis in range(3):
            A=eq.flux(eye,axis).T
            B=S@A@inverse
            torch.testing.assert_close(B,B.T)
            eigenvalues,Q=torch.linalg.eigh(B)
            torch.testing.assert_close(eigenvalues,torch.tensor([-eq.c,-eq.c,0,0,eq.c,eq.c],dtype=torch.float64),atol=1e-14,rtol=1e-13)
            absolute=inverse@(Q@torch.diag(eigenvalues.abs())@Q.T)@S
            exact=.5*(eq.flux(left,axis)+eq.flux(right,axis))-.5*(right-left)@absolute.T
            actual=eq.roe_flux(Face(left,eq.flux(left,axis)),Face(right,eq.flux(right,axis)),axis)
            torch.testing.assert_close(actual,exact,atol=2e-14,rtol=2e-13)
            same=Face(left,eq.flux(left,axis))
            torch.testing.assert_close(eq.roe_flux(same,same,axis),same.flux)

    def test_pec_mirror_and_analytic_walls(self):
        eq=Maxwell()
        U=torch.randn(5,6,dtype=torch.float64)
        case=MaxwellCavityCase(n=2)
        for axis in range(3):
            tangent=[a for a in range(3) if a!=axis]
            for side,coords in zip((-1,1),case.face_coords[axis]):
                ghost=eq.pec_boundary(coords,U,axis,side,.17)
                mean=.5*(U+ghost)
                torch.testing.assert_close(mean[...,tangent],torch.zeros_like(mean[...,tangent]))
                torch.testing.assert_close(mean[...,axis+3],torch.zeros_like(mean[...,axis+3]))
                analytic=case.mode.exact(*coords,.17)
                self.assertLess(analytic[...,tangent].abs().max().item(),2e-15)
                self.assertLess(analytic[...,axis+3].abs().max().item(),2e-15)

    def test_unsteady_maxwell_pde_and_divergence(self):
        mode=CavityMode(lengths=(1.1,1.6,2.3),mode=(2,1,3),epsilon=2.,mu=3.)
        x,y,z=torch.meshgrid(*(torch.linspace(.2,.7,3,dtype=torch.float64) for _ in range(3)),indexing='ij')
        coords=[x,y,z]
        t,eps=.13,1e-6
        residual=(mode.exact(*coords,t+eps)-mode.exact(*coords,t-eps))/(2*eps)
        divE=torch.zeros_like(x)
        divH=torch.zeros_like(x)
        for axis in range(3):
            plus,minus=coords.copy(),coords.copy()
            plus[axis]=plus[axis]+eps
            minus[axis]=minus[axis]-eps
            up,um=mode.exact(*plus,t),mode.exact(*minus,t)
            residual+=(mode.equation.flux(up,axis)-mode.equation.flux(um,axis))/(2*eps)
            divE+=(up[...,axis]-um[...,axis])/(2*eps)
            divH+=(up[...,axis+3]-um[...,axis+3])/(2*eps)
        self.assertLess(residual.abs().max().item(),2e-9)
        self.assertLess(divE.abs().max().item(),2e-9)
        self.assertLess(divH.abs().max().item(),2e-9)

    def test_semidiscrete_energy_central_and_roe(self):
        # Arbitrary discontinuous fields test the complete operator, including
        # all six physical walls, and weighted material energy, not just a mode.
        torch.manual_seed(7)
        for name in ('central','roe'):
            case=MaxwellCavityCase(cells=(2,3,1),p=2,mode=CavityMode(epsilon=2.3,mu=.7),numerical_flux=name)
            U=torch.randn(case.shape+(6,),dtype=torch.float64)
            rate=case.energy_rate(U)
            if name=='central':
                self.assertLess(abs(rate),1e-12)
            else:
                self.assertLess(rate,-1e-3)

    def test_analytic_energy_and_quarter_period(self):
        case=MaxwellCavityCase(n=3)
        self.assertAlmostEqual(case.energy(case.initial),case.mode.energy(),places=10)
        U,stats=case.run(case.mode.period/4,sample_every=5)
        self.assertLess(stats['field_l2'],.004)
        self.assertLess(abs(stats['relative_energy_change']),2e-5)
        electric=case.integral((U[...,:3]**2).sum(-1)[...,None]).item()*.5*case.equation.epsilon
        self.assertLess(electric/stats['energy'],1e-4)
        self.assertAlmostEqual(stats['magnetic_mode'],1.,delta=2e-4)
        self.assertLess(abs(stats['relative_frequency_error']),1e-3)
        torch.testing.assert_close(case.exact,case.mode.exact(*case.coords,case.mode.period/4))

    def test_spatial_convergence_both_fluxes(self):
        for name,minimum_ratio in (('central',6.),('roe',10.)):
            errors=[]
            for n in (2,4):
                case=MaxwellCavityCase(n=n,numerical_flux=name)
                _,stats=case.run(.1)
                errors.append(stats['field_l2'])
            self.assertGreater(errors[0]/errors[1],minimum_ratio)

    def test_vtk_and_history_output(self):
        case=MaxwellCavityCase(cells=(2,3,1))
        U,stats=case.run(.01,sample_every=1)
        with tempfile.TemporaryDirectory() as directory:
            stem=Path(directory)/'cavity'
            export(stem,case,U)
            root=ET.parse(stem.with_suffix('.vts')).getroot()
            piece=root.find('./StructuredGrid/Piece')
            self.assertEqual(piece.attrib['Extent'],'0 7 0 11 0 3')
            points=piece.find('Points/DataArray').text.split()
            self.assertEqual(len(points),8*12*4*3)
            # Verify VTK ordering using actual coordinates of the first two points.
            self.assertAlmostEqual(float(points[0]),case.coords[0][0,0,0,0,0,0].item())
            self.assertAlmostEqual(float(points[3]),case.coords[0][0,0,0,1,0,0].item())
            electric=piece.find("PointData/DataArray[@Name='E']").text.split()
            self.assertEqual(len(electric),8*12*4*3)
            with stem.with_name(stem.name+'_history.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(float(rows[0]['time']),0.)
            self.assertEqual(float(rows[-1]['time']),stats['time'])

    def test_invalid_parameters_and_custom_flux(self):
        for kwargs in ({'epsilon':0.},{'mu':float('nan')}):
            with self.assertRaises(ValueError):
                Maxwell(**kwargs)
        for kwargs in ({'lengths':(1.,2.,0.)},{'mode':(1,0,1)},{'amplitude':0.}):
            with self.assertRaises(ValueError):
                CavityMode(**kwargs)
        with self.assertRaises(ValueError):
            MaxwellCavityCase(numerical_flux='unknown')
        case=MaxwellCavityCase(n=1,numerical_flux=lambda minus,plus,axis:(minus.flux+plus.flux)/2)
        U,stats=case.run(0.)
        self.assertEqual(stats['field_l2'],0.)
        self.assertEqual(stats['flux'],'custom')


if __name__=='__main__':
    unittest.main()
