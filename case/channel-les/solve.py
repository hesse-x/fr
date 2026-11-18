"""Constant-mass-flux, coarse-grid channel LES with explicit SGS viscosity."""
from importlib import import_module
from pathlib import Path
from fr.les import CartesianLESMixin

_dns = import_module('case.channel-dns.long_run')


class ChannelLES(CartesianLESMixin, _dns.Channel):
    """DNS channel geometry and wall conditions with WALE by default."""

    def __init__(self, *args, sgs_model='wale', modal_filter='none',
                 filter_threshold=.1, filter_alpha=36., filter_order=2, **kwargs):
        if modal_filter not in ('none', 'hifiles'):
            raise ValueError('modal filter must be none or hifiles')
        super().__init__(*args, sgs_model=sgs_model, **kwargs)
        self.modal_filter = None
        if modal_filter == 'hifiles':
            from fr.modal_filter import CartesianModalFilter
            self.modal_filter = CartesianModalFilter(self.s, filter_threshold, filter_alpha, filter_order)

    def initial(self, amplitude=.5):
        state = super().initial(amplitude)
        return self.modal_filter.apply(state)[0] if self.modal_filter is not None else state

    def perturb(self, U, amplitude, event):
        state = super().perturb(U, amplitude, event)
        return self.modal_filter.apply(state)[0] if self.modal_filter is not None else state


def configure_parser(parser):
    parser.set_defaults(cfl=.3, re_tau=180.)
    parser.add_argument('--modal-filter', choices=('none', 'hifiles'), default='hifiles')
    parser.add_argument('--filter-threshold', type=float, default=.1)
    parser.add_argument('--filter-alpha', type=float, default=36.)
    parser.add_argument('--filter-order', type=int, default=2)
    parser.add_argument('--max-wall-y-plus', type=float, default=5.,
                        help='reference nearest-wall Gauss point limit (default: 5)')
    parser.add_argument('--sgs', dest='sgs_model', choices=('wale', 'vreman', 'smagorinsky', 'none'), default='wale')
    parser.add_argument('--cw', type=float, default=.5)
    parser.add_argument('--cv', type=float, default=.07)
    parser.add_argument('--cs', type=float, default=.1)
    parser.add_argument('--turbulent-prandtl', type=float, default=.9)
    parser.add_argument('--filter-ratio', type=float, default=1.)
    parser.add_argument('--filter-width', choices=('dof', 'half-dof', 'gauss-center'), default='dof')


def model_options(args):
    return {name: getattr(args, name) for name in
            ('sgs_model', 'cw', 'cv', 'cs', 'turbulent_prandtl', 'filter_ratio', 'filter_width',
             'modal_filter', 'filter_threshold', 'filter_alpha', 'filter_order')}


def main():
    _dns.main(model_class=ChannelLES, configure_parser=configure_parser,
              model_options=model_options, output_default=Path('results/channel-les'),
              default_mesh=Path(__file__).with_name('mesh')/'default.json', description=__doc__)


if __name__ == '__main__':
    main()
