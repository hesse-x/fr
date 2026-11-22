#!/usr/bin/env bash
set -euo pipefail

python_bin=python
if [[ -x venv/bin/python ]]; then
    python_bin=venv/bin/python
fi

export OMP_PROC_BIND=close
export OMP_PLACES='{0},{2},{4},{6},{8},{10},{12},{14}'
exec taskset -c 0,2,4,6,8,10,12,14 "$python_bin" -u -m case.round_jet_les "$@"
