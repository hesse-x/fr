"""Independent analytic checks for 2D convected FW-H propagation."""
import unittest
import numpy as np
from fwh import green,radiation,harmonic_fit,fit_frequency,sources


class FWHTests(unittest.TestCase):
    def test_conservative_flux_source_transformation(self):
        theta=np.linspace(0,2*np.pi,17,endpoint=False)
        n=np.column_stack((np.cos(theta),np.sin(theta)))
        mean=np.array([.2,.1]);rho=1+.01*np.cos(theta)
        velocity=mean+np.column_stack((.03*np.sin(theta),.02*np.cos(theta)))
        pressure=1/1.4+.003*np.sin(2*theta)
        stress_normal=.0001*np.column_stack((np.cos(theta),np.sin(theta)))
        normal_speed=np.sum(velocity*n,axis=-1)
        mass=rho*normal_speed
        momentum=mass[:,None]*velocity+pressure[:,None]*n-stress_normal
        q,f=sources(mass,momentum,n,mean)
        expected_q=np.sum((rho[:,None]*velocity-mean)*n,axis=-1)
        expected_f=rho[:,None]*(velocity-mean)*normal_speed[:,None]+(pressure-1/1.4)[:,None]*n-stress_normal-expected_q[:,None]*mean
        np.testing.assert_allclose(q,expected_q,atol=1e-15)
        np.testing.assert_allclose(f,expected_f,atol=1e-15)

    def test_green_derivative(self):
        xy=np.array([[3.,4.],[-5.,2.]])
        for mean in ((0.,0.),(.2,0.),(.1,.15)):
            g,gradient=green(xy,.7,mean)
            for axis in (0,1):
                delta=np.eye(2)[axis]*1e-5
                difference=(green(xy+delta,.7,mean)[0]-green(xy-delta,.7,mean)[0])/2e-5
                np.testing.assert_allclose(difference,gradient[:,axis],rtol=1e-8,atol=1e-10)

    def test_analytic_surface_reconstruction(self):
        omega=.6;rho0=1.;c0=1.
        observers=np.array([[30.,0.],[0.,30.],[-30.,5.],[20.,20.]])
        for mean in (np.array([0.,0.]),np.array([.2,0.]),np.array([.1,.15])):
            for radius in (1.,2.5):
                theta=np.arange(256)*2*np.pi/256
                normals=np.column_stack((np.cos(theta),np.sin(theta)))
                points=radius*normals;weights=np.full(256,2*np.pi*radius/256)
                for dipole in (False,True):
                    def field(xy):
                        g,grad=green(xy,omega,mean,c0)
                        if not dipole:return g,grad
                        delta=np.array([0.,1e-5])
                        hessian_y=(green(xy+delta,omega,mean,c0)[1]-green(xy-delta,omega,mean,c0)[1])/2e-5
                        return grad[:,1],hessian_y
                    phi,velocity=field(points)
                    pressure=rho0*(1j*omega*phi-velocity@mean)
                    un=normals@mean
                    q=rho0*np.sum(velocity*normals,axis=-1)+un*pressure/c0**2
                    f=rho0*un[:,None]*velocity+pressure[:,None]*normals-q[:,None]*mean
                    actual=radiation(points,normals,weights,q,f,observers,omega,mean,c0)
                    phi_o,v_o=field(observers)
                    expected=rho0*(1j*omega*phi_o-v_o@mean)
                    np.testing.assert_allclose(actual,expected,rtol=2e-7,atol=1e-9)

    def test_tonal_fit(self):
        times=np.arange(1200)*.25;frequency=.0313
        data=2+.0001*times+3*np.cos(2*np.pi*frequency*times)+4*np.sin(2*np.pi*frequency*times)
        fitted=fit_frequency(times,data,.2)
        self.assertAlmostEqual(fitted*.2,frequency,places=8)
        a=harmonic_fit(times,data[:,None],frequency,2)
        np.testing.assert_allclose(a[:,0],[3+4j,0],atol=1e-10)


if __name__=='__main__':unittest.main()
