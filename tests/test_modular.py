import math
import unittest
import torch
from fr import CartesianFR, Euler, integrate, rusanov_flux, ssprk43


torch.set_num_threads(1)


class ModularVerification(unittest.TestCase):
    def test_periodic_advection_all_dimensions(self):
        # Unequal cell counts and axis speeds expose permutation mistakes.
        for d in (1,2,3):
            cells=(3,4,2)[:d]
            velocity=(.7,-.3,.2)[:d]
            speed=lambda U,a: torch.full_like(U[...,0],abs(velocity[a]))
            s=CartesianFR(cells,((0.,1.),)*d,p=3,
                          flux=lambda U,a: velocity[a]*U,
                          numerical_flux=rusanov_flux(speed),periodic=True,speed=speed)
            U=(1+.1*sum(torch.sin(2*math.pi*x) for x in s.coords))[...,None]
            rhs=s.rhs(U)
            torch.testing.assert_close(s.integral(rhs),torch.zeros(1,dtype=U.dtype),atol=1e-13,rtol=0)
            V,stats=integrate(s.rhs,U,.02,dt=lambda U,t:s.timestep(U,.1))
            exact=(1+.1*sum(torch.sin(2*math.pi*(x-velocity[a]*.02)) for a,x in enumerate(s.coords)))[...,None]
            self.assertLess(s.norm(V-exact).max().item(),5e-4)
            torch.testing.assert_close(s.integral(V),s.integral(U),atol=1e-13,rtol=0)

    def test_time_dependent_boundaries_all_dimensions(self):
        # Exact affine solution U=2+sum(x_a)-d*t with F_a=U.
        for d in (1,2,3):
            s=CartesianFR((2,)*d,((0.,1.),)*d,p=1,
                          flux=lambda U,a: U,
                          flux_pair=lambda Fminus,Fplus: Fminus,
                          face_flux=lambda U,a: U,
                          boundary=lambda coords,inside,axis,side,t: (2+sum(coords)-d*t)[...,None])
            U=(2+sum(s.coords))[...,None]
            V,_=integrate(s.rhs,U,.1,dt=.03)
            torch.testing.assert_close(V,U-d*.1,atol=2e-13,rtol=0)

    def test_euler_free_stream_all_dimensions(self):
        for d in (1,2,3):
            e=Euler(d)
            s=CartesianFR((1,)*d,((0.,1.),)*d,p=2,flux=e.flux,
                          numerical_flux=rusanov_flux(e.speed),periodic=True,speed=e.speed)
            U=torch.zeros(s.shape+(d+2,),dtype=torch.float64)
            U[...,0]=1
            U[...,1:-1]=.2
            U[...,-1]=1/(e.gamma-1)+.5*d*.2**2
            torch.testing.assert_close(s.rhs(U),torch.zeros_like(U),atol=2e-13,rtol=0)
            self.assertTrue(e.valid(U))

    def test_ssprk43_order_nonautonomous(self):
        errors=[]
        for n in (10,20,40):
            U,_=integrate(lambda U,t: U+t,torch.tensor([1.],dtype=torch.float64),1.,dt=1/n)
            errors.append(abs(U.item()-(2*math.e-2)))
        self.assertGreater(math.log2(errors[0]/errors[1]),2.9)
        self.assertGreater(math.log2(errors[1]/errors[2]),2.9)
        times=[]
        ssprk43(lambda U,t: (times.append(t) or U),torch.ones(1),2.,.4)
        for actual,expected in zip(times,(2.,2.2,2.4,2.2)):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(len(times),4)

    def test_ssprk43_ssp_upwind(self):
        # First-order periodic upwind has dt_FE=h. SSP43 must preserve
        # positivity and total mass at dt=2*h, its SSP coefficient limit.
        s=CartesianFR((8,),((0.,1.),),p=0,flux=lambda U,a:U,
                      flux_pair=lambda minus,plus:minus,periodic=True)
        U=torch.zeros(s.shape+(1,),dtype=torch.float64)
        U[0]=1
        V=ssprk43(s.rhs,U,0.,2/8)
        self.assertGreaterEqual(V.min().item(),-1e-14)
        torch.testing.assert_close(s.integral(V),s.integral(U))

    def test_boundary_flux_is_recomputed_from_state(self):
        # For nonlinear F=u²/2, flux(trace(U)) != trace(flux(U)).
        # The face callback must see the former, and the FR correction must
        # still subtract the latter to conserve globally.
        seen=[]
        def face(U,a):
            seen.append(U.clone())
            return U*U/2
        s=CartesianFR((2,),((0.,1.),),p=2,flux=lambda U,a:U*U/2,
                      face_flux=face,flux_pair=lambda m,p:(m+p)/2,periodic=True)
        U=(1+s.coords[0])[...,None]
        rhs=s.rhs(U)
        self.assertGreater(len(seen),0)
        torch.testing.assert_close(s.integral(rhs),torch.zeros(1,dtype=U.dtype),atol=1e-13,rtol=0)
        right=s._trace(U,s.R)
        torch.testing.assert_close(seen[0],right[:-1])

    def test_rk4_regression(self):
        from case.ringleb import RinglebCase
        s=RinglebCase(4,3)
        _,stats=s.run(.1,method='rk4')
        self.assertAlmostEqual(stats['l2'][0]/2.221927903637049e-8,1.,places=5)


if __name__=='__main__':
    unittest.main()
