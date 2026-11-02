"""Generate the current NACA NS structured mesh with a 10-chord outer radius."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('naca_mesh_generator',ROOT/'case/naca0012-euler/generate_mesh.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
geometry=module.geometry


if __name__=='__main__':
    module.main(default_radius=10.,default_output=Path(__file__).parent/'results/mesh/structured.msh')
