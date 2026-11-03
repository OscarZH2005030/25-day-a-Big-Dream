"""Rebuild the forty synthetic input cases and compare their L1 repairs."""
from pathlib import Path

import stress_source_clip
from stress_observable_gate import load


if __name__ == '__main__':
    stress_source_clip.main()
    archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    load('registered_l1_repair',archive/'task4_l1_repair_stress.py').main()
