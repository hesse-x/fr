"""Durable LES logs, exit status and automatic mean-field export."""
from importlib import import_module
import sys

if __name__ == '__main__':
    sys.exit(import_module('case.channel-dns.run_logged').main(runner_module='case.channel-les'))
