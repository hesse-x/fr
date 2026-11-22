import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from case.round_jet_les.fwh_sampling import FWHRecorder, surface_quadrature
from case.round_jet_les.fwh import _observer_kernel, _time_transform
from fr import NavierStokes


class RoundJetFWHTests(unittest.TestCase):
    plan = dict(upstream_cap_x_D=.2, nominal_downstream_cap_x_D=.6,
                alternative_downstream_caps_x_D=[.4,.8],
                radius_coefficients=[.3,.1,.02], secondary_surface_radius_offset_D=.1)

    def test_closed_surfaces_share_nominal_upstream_disk(self):
        xyz, normal, area, patches, surfaces = surface_quadrature(
            self.plan, azimuth_count=16)
        self.assertEqual(len(surfaces), 4)
        self.assertEqual(surfaces['nominal_0.4D'][0], surfaces['nominal_0.8D'][0])
        self.assertEqual(len(xyz), len(area))
        np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.)
        self.assertTrue((area>0.).all())
        vectors = normal*area[:,None]
        for names in surfaces.values():
            np.testing.assert_allclose(sum(vectors[slice(*patches[name])].sum(0)
                                           for name in names), 0., atol=1e-13)

    def test_uniform_samples_resume_at_last_committed_frame(self):
        eq = NavierStokes(3)
        model = SimpleNamespace(eq=eq, pressure=2.8, mach=.5, mesh_digest='test',
                                edges=[0.,1.],
                                probes=lambda xyz: lambda state: state.expand(len(xyz[0]), 5))

        def state(time):
            return eq.conservative(torch.tensor([1.+time], dtype=torch.float64),
                                   torch.zeros(1,3,dtype=torch.float64),
                                   torch.tensor([model.pressure+time], dtype=torch.float64))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root/'plan.json'
            plan.write_text(json.dumps(self.plan))
            recorder = FWHRecorder(model, root/'fwh', plan, .04, 0., None)
            recorder.record(state(0.), 0.)
            recorder.record(state(.07), .07, state(0.), 0.)
            count, points = recorder.count, recorder.point_count
            recorder.close()
            self.assertEqual(count, 2)
            path = root/'fwh'/'surface.f32'
            with path.open('ab') as stream:
                stream.write(b'partial frame')
            recorder = FWHRecorder(model, root/'fwh', plan, .04, .07, count)
            recorder.record(state(.12), .12, state(.07), .07)
            recorder.close()
            data = np.fromfile(path, dtype='<f4').reshape(4, points, 5)
            for column in (0,4):
                np.testing.assert_allclose(data[:,0,column], [.0,.04,.08,.12], atol=3e-8)
            np.testing.assert_array_equal(data[:,:,1:4], 0.)

    def test_angular_projection_matches_direct_retarded_integral(self):
        azimuth, modes = 32, 3
        phi = 2*np.pi*np.arange(azimuth)/azimuth
        r, x, polar, distance, sound, st = 1.5, 2., .6, 50., 2., .7
        frequency = np.array([st])
        kernel = _observer_kernel(np.array([x]),np.array([r]),np.ones(1),azimuth,
                                  polar,distance,frequency,sound,modes)[0,0]
        harmonics = np.r_[0,np.arange(1,modes+1),-np.arange(1,modes+1)]
        rng = np.random.default_rng(27)
        spectra = rng.normal(size=(len(harmonics),4))+1j*rng.normal(size=(len(harmonics),4))
        for angle in (.2,2.3):
            values = np.exp(1j*phi[:,None]*harmonics)@spectra
            displacement = np.array([np.full_like(phi,distance*np.cos(polar)-x),
                                     distance*np.sin(polar)*np.cos(angle-phi)-r,
                                     distance*np.sin(polar)*np.sin(angle-phi)])
            ray = np.linalg.norm(displacement,axis=0)
            omega = 2j*np.pi*st
            green = np.exp(-omega*(ray-distance)/sound)/(4*np.pi*ray)
            direct = np.sum(green*(omega*values[:,0]+(omega/sound+1/ray)
                                  *np.einsum('pi,ip->p',values[:,1:],displacement)/ray))
            projected = np.sum(kernel*spectra*np.exp(1j*angle*harmonics)[:,None])
            np.testing.assert_allclose(projected,direct,rtol=2e-6,atol=1e-7)

    def test_temporal_transform_reconstructs_negative_azimuthal_modes(self):
        source = np.arange(15).reshape(1,1,3,5).astype(np.complex64)
        source[:,:,1:] += 1j*source[:,:,1:,::-1]
        actual = _time_transform(source,10,5)[:,0,:,0]
        both = np.concatenate((source,source[:,:,1:].conj()),axis=2)
        expected = np.fft.fft(both,n=10,axis=-1)[0,0,:,:5].T
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
