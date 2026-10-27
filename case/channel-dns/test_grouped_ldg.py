import unittest
import torch
from channel import Channel
from grouped_convection import GroupedConvection
from grouped_ldg import GroupedLDG
from tensor_ops import ssprk43_add


class GroupedLDGTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_halo_walls_and_nonuniform_state(self):
        for cells, block in (((1, 1, 1), (2, 2, 2)), ((2, 3, 2), (1, 2, 1)),
                             ((5, 4, 3), (2, 3, 2))):
            with self.subTest(cells=cells, block=block):
                model = Channel(cells, stretch=1.7,)
                state = model.perturb(model.initial(), .4, 3)
                generator = torch.Generator().manual_seed(42)
                density = .8+.4*torch.rand(state.shape[:-1]+(1,), generator=generator, dtype=state.dtype)
                state = state*density
                x, y, z = model.s.coords
                delta = .03*model.temperature*(1-y*y)*torch.cos(x)*torch.cos(2*z)
                state[..., 4] += state[..., 0]*delta/(model.eq.gamma-1)
                state = state.movedim(-1, 0).contiguous().movedim(0, -1)
                convection = GroupedConvection(model, block)
                ldg = GroupedLDG(model, block)
                actual = ldg(state, convection(state))
                torch.testing.assert_close(actual, model.rhs(state), atol=1e-8, rtol=1e-10)
                self.assertEqual(actual.stride(), state.stride())
                self.assertLess(abs(float(model.s.integral(actual)[0])), 1e-10)

    def test_quadratic_heat_conduction(self):
        model = Channel((3, 5, 2), mach=.5, forcing=0., stretch=1.7,)
        _, y, _ = model.s.coords
        temperature = model.temperature+20*(1-y*y)
        velocity = torch.zeros((*model.s.shape, 3), dtype=torch.float64)
        state = model.state(velocity, temperature[..., None])
        state = state.movedim(-1, 0).contiguous().movedim(0, -1)
        ldg = GroupedLDG(model, (2, 3, 1))
        actual = ldg(state, torch.zeros_like(state))
        self.assertLess(float(actual[..., :4].abs().max()), 1e-9)
        # rho=1, velocity=0: energy diffusion is kappa * Laplacian(T).
        self.assertLess(float((actual[..., 4]+40*model.kappa).abs().max()), 1e-8)

    def test_multistep_and_rest(self):
        model = Channel((4, 5, 3), stretch=1.7,)
        state = model.perturb(model.initial(), .4, 3).movedim(-1, 0).contiguous().movedim(0, -1)
        convection = GroupedConvection(model, (2, 3, 2))
        ldg = GroupedLDG(model, (2, 3, 2))
        rhs = ldg.make_rhs(convection)
        dt = model.timestep(state)
        reference, actual = state, state
        for i in range(16):
            reference = ssprk43_add(model.rhs, reference, i*dt, dt)
            actual = ssprk43_add(rhs, actual, i*dt, dt)
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-12)
        self.assertTrue(model.eq.valid(actual))
        zero = torch.zeros((*model.s.shape, 3), dtype=torch.float64)
        rest = model.state(zero).movedim(-1, 0).contiguous().movedim(0, -1)
        torch.testing.assert_close(rhs(rest), model.rhs(rest), atol=1e-8, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
