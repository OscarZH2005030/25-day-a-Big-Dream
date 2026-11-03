"""Rebuild the net projections and reproduce the diagnostic L1 programs."""
from pathlib import Path

import audit_task4_net_repair
from stress_observable_gate import load


if __name__ == '__main__':
    audit_task4_net_repair.main()
    archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    load('registered_minimum_invisible',archive/'task4_minimum_invisible_repair.py').main()
