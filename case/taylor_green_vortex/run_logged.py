"""Persist solver logs and exit status; generate reference comparison on success.

Accepts the same arguments as ``python -m case.taylor_green_vortex``.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main(module='case.taylor_green_vortex',
         default_output='results/taylor_green_vortex/ma03-n64-p3-t10-stats100'):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, default=Path(default_output))
    args, _ = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    command = [sys.executable, '-u', '-m', module, *sys.argv[1:]]
    record = dict(command=command, started_utc=stamp, parent_pid=os.getpid(), cwd=os.getcwd(), cpu_affinity=sorted(os.sched_getaffinity(0)))
    (args.output/f'process-{stamp}.json').write_text(json.dumps(record, indent=2)+'\n')
    log_path = args.output/f'run-{stamp}.log'
    print(f'Log: {log_path}', flush=True)
    with log_path.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat(), log=str(log_path))
    (args.output/f'exit-{stamp}.json').write_text(json.dumps(record, indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f'Solver finished; comparison is in {args.output / "comparison"}', flush=True)


if __name__ == '__main__':
    main()
