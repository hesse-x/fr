"""Persistent process log and exit status for the LES case."""
from case.taylor_green_vortex.run_logged import main

if __name__ == '__main__':
    main(module='case.taylor_green_vortex_les', default_output='results/taylor_green_vortex_les')
