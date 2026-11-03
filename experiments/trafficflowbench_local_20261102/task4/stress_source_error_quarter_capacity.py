"""Reproduce the full and quarter source amplitude comparison."""
from pathlib import Path
import runpy

from stress_observable_gate import load


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    runpy.run_path(str(root / 'stress_source_error_symmetric_capacity.py'), run_name='__main__')
    archive = root.parent / 'archive/daytime'
    load('audit_source_error_quarter_capacity', archive / 'audit_source_error_quarter_capacity.py').main()
    for script in ('verify_source_error_quarter_capacity', 'compare_source_error_capacity_amplitudes'):
        runpy.run_path(str(archive / (script + '.py')), run_name='__main__')
