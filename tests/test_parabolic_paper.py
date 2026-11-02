import unittest

import numpy as np

from case.parabolic_antenna.paper_compare import REFERENCE, beamwidth


class PaperComparisonTests(unittest.TestCase):
    def test_half_power_width_known_gaussian_and_scale(self):
        theta = np.linspace(-180, 180, 7201)
        power = np.exp(-theta**2/(2*10**2))
        expected = 2*np.sqrt(2*np.log(2))*10
        self.assertAlmostEqual(beamwidth(theta, power), expected, places=3)
        self.assertAlmostEqual(beamwidth(theta, 3*power), expected, places=3)
        self.assertIsNone(beamwidth(theta, np.ones_like(theta)))

    def test_digitized_reference_main_lobe_calibration(self):
        reference = np.genfromtxt(REFERENCE, delimiter=',', names=True)
        theta = reference['theta_degrees']
        self.assertTrue(bool((np.diff(theta) > 0).all()))
        for plane, lower, upper in (('E', 34., 40.), ('H', 28., 35.)):
            for method in ('vfem', 'meep'):
                db = reference[f'{plane}_{method}_db']
                valid = np.isfinite(db)
                self.assertGreater(valid.sum(), 300)
                self.assertLess(abs(db[valid].max()), .5)
                width = beamwidth(theta[valid], 10**(db[valid]/10))
                self.assertGreater(width, lower)
                self.assertLess(width, upper)
                # Both plotted curves peak near the paper's stated boresight.
                self.assertLess(abs(theta[valid][db[valid].argmax()]), 4.)


if __name__ == '__main__':
    unittest.main()
