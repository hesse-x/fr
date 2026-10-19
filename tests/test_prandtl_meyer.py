import math
import unittest
import torch
from case.prandtl_meyer import PrandtlMeyerFan, PrandtlMeyerCase, prandtl_meyer


torch.set_num_threads(1)


class PrandtlMeyerVerification(unittest.TestCase):
    def test_nasa_mach25_turn15_reference(self):
        ref=PrandtlMeyerFan().reference()
        # NASA NPARC pm15, Table 7 (exact values rounded to four decimals).
        for name,value in (('mach2',3.2368),('pressure_ratio',.3274),
                           ('density_ratio',.4505),('temperature_ratio',.7269)):
            self.assertAlmostEqual(ref[name],value,delta=6e-5)

    def test_isentropic_invariants_and_uniform_regions(self):
        fan=PrandtlMeyerFan()
        angles=torch.tensor([fan.tail-.01,(fan.tail+fan.head)/2,fan.head+.01],dtype=torch.float64)
        x=torch.ones_like(angles)
        y=angles.tan()
        rho,velocity,p,M=fan.fields(x,y)
        theta=torch.atan2(velocity[...,1],velocity[...,0])
        torch.testing.assert_close(theta+prandtl_meyer(M),torch.full_like(M,fan.nu1),atol=1e-13,rtol=0)
        torch.testing.assert_close(p/rho**fan.gamma,torch.full_like(p,fan.pressure1/fan.rho1**fan.gamma))
        enthalpy=fan.gamma/(fan.gamma-1)*p/rho+.5*(velocity**2).sum(-1)
        torch.testing.assert_close(enthalpy,enthalpy[-1].expand_as(enthalpy))
        self.assertAlmostEqual(M[0].item(),fan.mach2)
        self.assertAlmostEqual(M[-1].item(),fan.mach1)
        self.assertAlmostEqual(theta[0].item(),-fan.turn)
        self.assertLess(p[0].item(),p[-1].item())
        for angle in (fan.head,fan.tail):
            yl=torch.full((1,),math.tan(angle-1e-9),dtype=torch.float64)
            yr=torch.full((1,),math.tan(angle+1e-9),dtype=torch.float64)
            torch.testing.assert_close(fan.exact(torch.ones_like(yl),yl),fan.exact(torch.ones_like(yr),yr),atol=1e-7,rtol=0)

    def test_continuous_euler_residual(self):
        # Independent Cartesian finite differences away from fan-edge kinks.
        for gamma in (1.4,1.67):
            fan=PrandtlMeyerFan(gamma=gamma)
            x=torch.linspace(.6,1.5,9,dtype=torch.float64)
            angle=torch.linspace(fan.tail+.02,fan.head-.02,9,dtype=torch.float64)
            y=x*angle.tan()
            e=fan.equation
            eps=1e-5
            div=(e.flux(fan.exact(x+eps,y),0)-e.flux(fan.exact(x-eps,y),0))/(2*eps)
            div+=(e.flux(fan.exact(x,y+eps),1)-e.flux(fan.exact(x,y-eps),1))/(2*eps)
            self.assertLess(div.abs().max().item(),3e-7)

    def test_smooth_fourth_order(self):
        errors=[]
        for n in (3,6):
            case=PrandtlMeyerCase(n=n,p=3,region='smooth')
            phi=torch.atan2(case.y,case.x)
            self.assertTrue(((phi>case.fan.tail)&(phi<case.fan.head)).all())
            _,stats=case.run(.01)
            errors.append(stats['l2'][0])
        self.assertGreater(errors[0]/errors[1],12.)

    def test_full_fan_evolves_physically(self):
        case=PrandtlMeyerCase(n=4,p=3)
        U,stats=case.run(.02)
        self.assertTrue(case.equation.valid(U))
        self.assertGreater(stats['steps'],0)
        self.assertGreater(stats['mach_l2'],0)
        self.assertLess(stats['mach_l2'],.05)
        phi=torch.atan2(case.y,case.x)
        self.assertTrue((phi<case.fan.tail).any())
        self.assertTrue((phi>case.fan.head).any())

    def test_csv_svg_use_case_gamma_and_bounds(self):
        import csv
        import tempfile
        from pathlib import Path
        import xml.etree.ElementTree as ET
        from fr.output import export
        case=PrandtlMeyerCase(n=2,fan=PrandtlMeyerFan(gamma=1.67))
        with tempfile.TemporaryDirectory() as directory:
            stem=Path(directory)/'pm'
            export(stem,case,case.exact)
            with stem.with_suffix('.csv').open() as f:
                row=next(csv.DictReader(f))
            point=case.fan.fields(case.x.flatten()[:1],case.y.flatten()[:1])
            self.assertAlmostEqual(float(row['Mach']),point[-1].item(),places=12)
            self.assertEqual(float(row['abs_density_error']),0.)
            root=ET.parse(stem.with_suffix('.svg')).getroot()
            text=' '.join(root.itertext())
            self.assertIn('x: [0.5, 1.5]',text)

    def test_invalid_parameters(self):
        for kwargs in ({'mach':1.},{'turn_degrees':0.},{'turn_degrees':89.,'mach':10.},
                       {'gamma':1.},{'rho':-1.},{'mach':float('nan')}):
            with self.assertRaises(ValueError):
                PrandtlMeyerFan(**kwargs)
        fan=PrandtlMeyerFan()
        with self.assertRaises(ValueError):
            fan.exact(torch.tensor([0.]),torch.tensor([0.]))
        with self.assertRaises(ValueError):
            fan.exact(torch.tensor([1.]),torch.tensor([-1.]))


if __name__=='__main__':
    unittest.main()
