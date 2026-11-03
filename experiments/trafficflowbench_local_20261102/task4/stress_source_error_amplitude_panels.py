"""Summarize road-level sensitivity from a reproduced quarter-amplitude run."""
import argparse
from pathlib import Path
import sys
import types

from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    root = parser.parse_args().output.resolve()
    common = types.ModuleType('common')
    common.OUT = root
    common.sha = sha
    common.save = save
    common.state = lambda **unused: None
    sys.modules['common'] = common
    archive = Path(__file__).resolve().parents[1] / 'archive/daytime'
    load('audit_source_error_amplitude_panels', archive / 'audit_source_error_amplitude_panels.py').main()


if __name__ == '__main__':
    main()
