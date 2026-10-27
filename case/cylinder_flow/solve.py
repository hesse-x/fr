"""Run cylinder flow with the shared explicit laminar Navier–Stokes driver."""
import importlib.util
import argparse
from pathlib import Path
import sys

from cylinder import ROOT
from native import NativeCylinder


class Observers:
    def __init__(self,items):self.items=items
    def __call__(self,U,t):
        for item in self.items:item(U,t)
    def finish(self,U,t):
        for item in self.items:item.finish(U,t)


def main():
    plotting=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    plotting.add_argument('--frames',type=int,default=120)
    plotting.add_argument('--fps',type=int,default=15)
    plotting.add_argument('--plot-start',type=float,default=0.)
    plotting.add_argument('--vorticity-limit',type=float,default=3.)
    plotting.add_argument('--no-plot',action='store_true')
    plotting.add_argument('--fwh-output',type=Path,help='write closed-contour ASCII samples for fwh.py')
    plotting.add_argument('--fwh-start',type=float,default=0.,help='first FW-H sample, convective time')
    plotting.add_argument('--fwh-interval',type=float,default=.05,help='uniform FW-H sample interval in convective time')
    options,remaining=plotting.parse_known_args()
    if '-h' in remaining or '--help' in remaining:
        plotting.print_help()
    sys.argv[1:]=remaining
    # The existing driver imports `airfoil`; supply its directory explicitly,
    # then replace the physics class with the diameter-normalized cylinder.
    ns_dir = ROOT/'case/naca0012-ns'
    sys.path.insert(0, str(ns_dir))
    try:
        spec = importlib.util.spec_from_file_location('cylinder_ns_driver', ns_dir/'solve.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
    finally:
        sys.path.remove(str(ns_dir))
    driver.Airfoil = NativeCylinder
    base = Path(__file__).resolve().parent/'results'
    defaults = {'--mesh': str(base/'mesh/structured.msh'),
                '--output': str(base/'re100'), '--re': '100',
                '--time': '100', '--max-steps': '1000000',
                '--residual-tol': '1e-30', '--threads': '4'}
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag+'=') for arg in sys.argv[1:]):
            sys.argv.extend([flag, value])
    def factory(physics,args,t):
        observers=[]
        if not options.no_plot:
            from visualize import WakeRecorder
            observers.append(WakeRecorder(physics,args,frames=options.frames,
                fps=options.fps,start=max(options.plot_start,t*physics.mach),limit=options.vorticity_limit))
        if options.fwh_output:
            from fwh_sampling import FWHRecorder
            observers.append(FWHRecorder(physics,args,options.fwh_output,
                start=max(options.fwh_start,t*physics.mach),interval=options.fwh_interval))
        return Observers(observers) if observers else None
    driver.main(observer_factory=factory)


if __name__ == '__main__':
    main()
