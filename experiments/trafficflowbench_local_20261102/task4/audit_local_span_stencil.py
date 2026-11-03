"""Rebuild the frozen anchors and reproduce the local integer stencil audit."""
from pathlib import Path

import audit_extra_measurements
from stress_observable_gate import load


if __name__ == '__main__':
    audit_extra_measurements.main()
    source = Path(__file__).resolve().parents[1]/'archive/daytime/audit_local_span_null_stencil.py'
    load('registered_span_stencil',source).main()
