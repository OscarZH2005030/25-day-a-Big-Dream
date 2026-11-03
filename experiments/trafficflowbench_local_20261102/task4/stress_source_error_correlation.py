"""Rebuild and verify the matched source error alignment experiment."""
from pathlib import Path
import runpy

from stress_observable_gate import load


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    runpy.run_path(str(root / 'stress_source_agreement_common_corruption.py'), run_name='__main__')
    archive = root.parent / 'archive/daytime'
    load('audit_source_error_correlation', archive / 'audit_source_error_correlation.py').main()
    runpy.run_path(str(archive / 'verify_source_error_correlation.py'), run_name='__main__')
