"""Curved triangular NS starter mesh with wall and wake refinement."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec = importlib.util.spec_from_file_location('euler_mesh', ROOT/'case/naca0012-euler/generate_mesh.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.geometry


def geometry(radius=10.,wall_size=.04,far_size=2.,downstream=20.):
    geo = original(radius,wall_size,far_size,downstream)
    return geo.replace('Background Field=2;', '\n'.join([
        'Field[3]=Box;', f'Field[3].VIn={2*wall_size};', f'Field[3].VOut={far_size};',
        'Field[3].XMin=0.8;', 'Field[3].XMax=4;', 'Field[3].YMin=-0.15;', 'Field[3].YMax=0.15;',
        'Field[3].Thickness=0.3;', 'Field[4]=Min;', 'Field[4].FieldsList={2,3};', 'Background Field=4;']))


if __name__ == '__main__':
    # Delegate CLI and validation while supplying the NS geometry and defaults.
    module.geometry = geometry
    defaults = {'--output':str(Path(__file__).parent/'results/mesh/coarse.msh'),
                '--wall-size':'.04','--radius':'10','--far-size':'2'}
    for flag,value in defaults.items():
        if not any(a==flag or a.startswith(flag+'=') for a in sys.argv[1:]):
            sys.argv.extend([flag,value])
    module.main()
