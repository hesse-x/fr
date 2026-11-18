"""Migrated filter: native/Torch parity, mean preservation and RK ordering."""
from importlib import import_module
import unittest
import torch
from fr.modal_filter import CartesianModalFilter
from fr.native.cartesian_ns.modal_filter import NativeCartesianModalFilter
from fr.native.cartesian_ns.native_step import NativeMeanAccumulator

ChannelLES = import_module('case.channel-les').ChannelLES


class CartesianModalFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **kwargs):
        return ChannelLES((2, 6, 2), mach=.3, forcing=0., modal_filter='hifiles', **kwargs)

    def oscillatory(self, model):
        U = model.initial()
        z = model.s.z.reshape(1, 1, 1, 4, 1, 1)
        U[..., 0] += .2*(5*z**3-3*z)/2
        U[..., 4] += .1*z
        return U

    def test_native_reference_and_cellwise_physical_integral(self):
        model = self.model()
        U = self.oscillatory(model)
        expected, sensor, mask = model.modal_filter.apply(U)
        self.assertTrue(bool(mask.all()))
        native = NativeCartesianModalFilter(model)
        actual = model.soa(U)
        native.apply(actual)
        torch.testing.assert_close(actual, expected, rtol=3e-13, atol=3e-12)
        torch.testing.assert_close(native.cell_sensor.reshape(model.s.cells), sensor, rtol=3e-13, atol=3e-13)
        w = model.s.weights[..., None]
        torch.testing.assert_close((actual*w).sum((3, 4, 5)), (U*w).sum((3, 4, 5)),
                                   rtol=3e-13, atol=3e-11)
        torch.testing.assert_close(model.s.integral(actual), model.s.integral(U), rtol=3e-13, atol=3e-10)
        self.assertLess(native.summary()['maximum_mean_error'], 1e-10)

    def test_constants_and_small_density_gradient_unchanged(self):
        model = self.model()
        native = NativeCartesianModalFilter(model)
        for gradient in (0., .001):
            U = model.initial()
            U[..., 0] += gradient*model.s.z.reshape(1, 1, 1, 1, 4, 1)
            expected, _, mask = model.modal_filter.apply(U)
            self.assertFalse(bool(mask.any()))
            torch.testing.assert_close(expected, U, rtol=0, atol=0)
            actual = model.soa(U)
            native.apply(actual)
            torch.testing.assert_close(actual, U, rtol=0, atol=0)

    def test_every_rk_stage_and_filtered_statistics(self):
        model = self.model(filter_threshold=.001)
        U = model.soa(self.oscillatory(model))
        U = model.soa(model.modal_filter.apply(U)[0])
        copy = U.clone()
        native = model.make_stepper('native-p3', constant_mass_flux=True)
        expected = model.make_stepper('torch', constant_mass_flux=True)(U, 0., 1e-5)
        integral = NativeMeanAccumulator(model).zeros()
        actual, valid, dt = native.advance_finish(U, 0., 1e-5, .3, integral)
        self.assertTrue(valid)
        self.assertEqual(native.modal_filter.calls, 4)
        torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-11)
        torch.testing.assert_close(U, copy, rtol=0, atol=0)
        expected_integral = 1e-5/2*(model.field_moments(U)+model.field_moments(actual))
        torch.testing.assert_close(integral, expected_integral, rtol=3e-13, atol=3e-13)
        self.assertAlmostEqual(dt, model.timestep(actual, .3), places=13)
        torch.testing.assert_close(model.s.integral(actual)[1], model.s.integral(U)[1], rtol=3e-13, atol=3e-11)

    def test_density_sensor_is_not_a_pressure_limiter(self):
        model = self.model()
        U = torch.zeros_like(model.initial())
        U[..., 0] = 1.
        U[..., 1] = 1.4+.9*model.s.z.reshape(1, 1, 1, 4, 1, 1)
        U[..., 4] = 2.5
        self.assertTrue(model.eq.valid(U))
        actual, _, mask = model.modal_filter.apply(U)
        self.assertFalse(bool(mask.any()))
        face = (actual.movedim(3, -2)*model.s.R[:, None]).sum(-2)
        self.assertTrue(bool((model.eq.primitive(face)[2] < 0).any()))
        torch.testing.assert_close(actual, U, rtol=0, atol=0)

    def test_invalid_filter_parameters(self):
        model = self.model()
        for kw in ({'threshold': -1.}, {'alpha': 0.}, {'order': 3}):
            with self.assertRaises(ValueError):
                CartesianModalFilter(model.s, **kw)


if __name__ == '__main__':
    unittest.main()
