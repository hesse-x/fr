import itertools
import unittest
import torch
from numpy.polynomial import Polynomial

from fr import CartesianFR, Euler, central, rusanov_flux
from fr.operators import overintegration


torch.set_num_threads(1)
MODES = ('none', 'flux', 'surf-flux', 'flux, surf-flux')


class DealiasingVerification(unittest.TestCase):
    def test_projection_not_interpolation(self):
        q, _, I, P = overintegration(2, 5)
        from fr.operators import operators
        z = operators(2, 'cpu')[0]
        torch.testing.assert_close(P @ q**4, 6/7*z**2-3/35, atol=2e-14, rtol=0)
        torch.testing.assert_close(P @ I, torch.eye(3, dtype=z.dtype), atol=2e-14, rtol=0)
        self.assertGreater((P @ q**4-z**4).abs().max().item(), .05)

    def test_quadratic_flux_exact_weak_residual(self):
        # Independent polynomial integration of the DG weak form, U=x^3.
        s = CartesianFR((1,), ((-1., 1.),), p=3, flux=lambda U,a: U**2,
                        numerical_flux=central, periodic=True, anti_alias='flux')
        z = s.z.numpy()
        expected = []
        for i in range(4):
            basis = Polynomial([1.])
            for j in range(4):
                if j != i:
                    basis *= Polynomial([-z[j], 1.])/(z[i]-z[j])
            primitive = (basis.deriv()*Polynomial([0,0,0,0,0,0,1])).integ()
            expected.append((primitive(1)-primitive(-1)-basis(1)+basis(-1))/s.w[i].item())
        U = s.coords[0][...,None]**3
        torch.testing.assert_close(s.rhs(U).flatten(), torch.tensor(expected, dtype=U.dtype), atol=3e-14, rtol=0)
        raw = CartesianFR((1,), ((-1.,1.),), p=3, flux=lambda U,a: U**2,
                          numerical_flux=central, periodic=True)
        self.assertGreater((raw.rhs(U)-s.rhs(U)).abs().max().item(), .1)

    def test_linear_equivalence_all_dimensions_orders_and_modes(self):
        for d,p in itertools.product((1,2,3), (0,2)):
            kwargs = dict(cells=(2,3,1)[:d], bounds=((0.,1.),)*d, p=p,
                          flux=lambda U,a: (a+1)*U, numerical_flux=central, periodic=True)
            base = CartesianFR(**kwargs)
            U = (1+sum(x*x for x in base.coords))[...,None]
            expected = base.rhs(U)
            for mode in MODES:
                s = CartesianFR(**kwargs, anti_alias=mode)
                torch.testing.assert_close(s.rhs(U), expected, atol=2e-12, rtol=2e-12)

    def test_euler_conservation_nonuniform_all_modes(self):
        for d,mode in itertools.product((1,2,3), MODES):
            e = Euler(d)
            s = CartesianFR((2,)*d, ((0.,1.),)*d, p=2, flux=e.flux,
                            numerical_flux=rusanov_flux(e.speed), periodic=True,
                            edges=([0.,.3,1.],)*d, anti_alias=mode)
            U = torch.zeros(s.shape+(d+2,), dtype=torch.float64)
            U[...,0] = 1+.05*sum(torch.sin(5*x) for x in s.coords)
            U[...,1:-1] = .2
            U[...,-1] = 3+.1*sum(torch.cos(3*x) for x in s.coords)
            torch.testing.assert_close(s.integral(s.rhs(U)), torch.zeros(d+2,dtype=U.dtype), atol=3e-13, rtol=0)

    def test_boundary_coordinates_and_integrated_flux(self):
        # Exact polynomial state: boundary callback must receive refined coordinates.
        for d,mode in itertools.product((2,3), MODES[1:]):
            calls=[]
            def boundary(coords, inside, axis, side, t):
                value=(2+sum(x*x for x in coords)+t)[...,None]
                torch.testing.assert_close(inside,value,atol=3e-13,rtol=0)
                calls.append(inside.shape)
                return value
            s=CartesianFR((1,)*d, ((0.,1.),)*d, p=2, flux=lambda U,a: U**2,
                          numerical_flux=central, boundary=boundary, anti_alias=mode)
            U=(2+sum(x*x for x in s.coords)+.1)[...,None]
            # Net outward flux per direction is integral [2*(2.1+sum x_t^2)+1].
            expected=-d*(5.2+2*(d-1)/3)
            self.assertAlmostEqual(s.integral(s.rhs(U,.1)).item(),expected,places=11)
            self.assertEqual(len(calls),2*d)
            if 'surf-flux' in mode:
                self.assertEqual(calls[0][-2],s.quadrature_order)

    def test_invalid_configuration(self):
        for option in ('typo', '', 'none, flux', True):
            with self.assertRaises(ValueError):
                CartesianFR((1,), ((0.,1.),), flux=lambda U,a: U,
                            numerical_flux=central, periodic=True, anti_alias=option)
        for n in (0,3,4.5):
            with self.assertRaises(ValueError):
                CartesianFR((1,), ((0.,1.),), p=3, flux=lambda U,a: U,
                            numerical_flux=central, periodic=True, quadrature_order=n)

    def test_nonlinear_tensor_quadrature_convergence(self):
        # Cubic flux of Q3 states times Q3 tests has degree <=12 per axis.
        # Seven Gauss points integrate it exactly; ten give an independent rule.
        for mode in MODES[1:]:
            kwargs=dict(cells=(2,1), bounds=((-1.,1.),)*2, p=3,
                        flux=lambda U,a: (a+1)*U**3,
                        numerical_flux=central, periodic=True)
            s=CartesianFR(**kwargs,anti_alias=mode,quadrature_order=7)
            ref=CartesianFR(**kwargs,anti_alias=mode,quadrature_order=10)
            x,y=s.coords
            U=(1+.3*x**3+.4*y**3+.1*x*y)[...,None]
            torch.testing.assert_close(s.rhs(U),ref.rhs(U),atol=4e-12,rtol=4e-12)
            raw=CartesianFR(**kwargs)
            self.assertGreater((s.rhs(U)-raw.rhs(U)).abs().max().item(),1e-3)


if __name__ == '__main__':
    unittest.main()
