"""Alternate fused/split-boundary complete-step benchmarks in fresh processes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cells', type=int, nargs=3, default=[64, 24, 32])
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, mode in enumerate((0, 1, 1, 0, 0, 1)):
        tag = f'{i:02d}-'+('split' if mode else 'fused')
        target = args.output/(tag+'.json')
        command = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
                   '--warmup', str(args.warmup), '--steps', str(args.steps),
                   '--repeats', str(args.repeats), '--threads', str(args.threads),
                   '--cells', *map(str, args.cells), '--output', str(target),
                   '--boundary-mode', 'separate' if mode else 'fused']
        print('START', tag, flush=True)
        with (args.output/(tag+'.log')).open('w') as log:
            subprocess.run(command, env=os.environ.copy(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(target.read_text())
        records.append(dict(mode=mode, file=target.name, **result))
        print(tag, result['seconds_per_step'], flush=True)
    summary = summarize(records)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


def summarize(records):
    # Pair adjacent runs: AB, BA, AB to reduce ordering bias.
    pairs = []
    for i in range(0, len(records), 2):
        pair = {r['mode']: r for r in records[i:i+2]}
        fused, split = pair[0]['seconds_per_step'], pair[1]['seconds_per_step']
        pairs.append(dict(fused=fused, split=split, slowdown_percent=100*(split/fused-1)))
    medians = {name: statistics.median(r['seconds_per_step'] for r in records if r['mode']==mode)
               for mode, name in ((0, 'fused'), (1, 'split'))}
    return dict(seconds_per_step=medians,
                slowdown_percent=100*(medians['split']/medians['fused']-1),
                pairs=pairs,
                minor_faults=sum(s['minor_faults'] for r in records for s in r['samples']),
                files=[r['file'] for r in records])


if __name__ == '__main__':
    main()
