"""Rebuild the hypothetical inverse cases and reproduce the physical projections."""
from pathlib import Path

import audit_hypothetical_time_conditioning
from stress_observable_gate import load


if __name__ == '__main__':
    audit_hypothetical_time_conditioning.main()
    archive = Path(__file__).resolve().parents[1]/'archive/daytime'
    load('audit_hypothetical_time_conditioning',archive/'audit_hypothetical_time_conditioning.py')
    load('registered_time_monotone',archive/'audit_hypothetical_time_monotone.py').main()
