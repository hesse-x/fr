"""Run long_run.py with durable stdout/stderr and signal/exit reporting."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = out/f'run-{stamp}.log'
    command = [sys.executable, '-u', '-X', 'faulthandler',
               str(Path(__file__).with_name('long_run.py')), '--output', str(out), *remaining]
    with path.open('a', buffering=1) as log:
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        metadata = dict(supervisor_pid=os.getpid(), pid=child.pid, log=str(path),
                        command=command, started=stamp)
        (out/'process.json').write_text(json.dumps(metadata, indent=2)+'\n')
        code = child.wait()
        metadata.update(returncode=code, signal=signal.Signals(-code).name if code < 0 else None,
                        finished=datetime.datetime.now().isoformat())
        log.write(f'calculation exited returncode={code} signal={metadata["signal"]}\n')
        (out/f'exit-{stamp}.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if code == 0:
            env = dict(os.environ)
            env.setdefault('MPLCONFIGDIR', '/tmp/fr-matplotlib')
            post = subprocess.run([sys.executable, str(Path(__file__).with_name('postprocess.py')),
                                   str(out/'checkpoint.pt')], cwd=root, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
            log.write(f'postprocess exited returncode={post.returncode}\n')
            metadata['postprocess_returncode'] = post.returncode
            (out/f'exit-{stamp}.json').write_text(json.dumps(metadata, indent=2)+'\n')
        log.flush()
        os.fsync(log.fileno())
    return 0 if code == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
