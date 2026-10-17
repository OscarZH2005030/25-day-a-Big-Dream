from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import (JOIN_KEY, day_files, eligible, network, read_day,
                   save_json, sha256, template)
from .pipeline import assert_hidden, panel_families, select_rows


def audit(args):
    root = args.release_root.resolve()
    families = panel_families(root)
    panels = list(families) if args.panels == ['all'] else args.panels
    report = {'release_root': str(root), 'sampled_days_per_split': args.max_days,
              'contract_sha256': sha256(root / 'config' / 'competition_contract.json'),
              'panels': [], 'note': 'Sample-level checks; file-calendar coverage is checked for every day.'}
    for panel in panels:
        net = network(root, panel)
        for split, start, end in [('train', '2030-06-01', '2031-02-28'),
                                  ('validation', '2031-03-01', '2031-03-31'),
                                  ('private', '2031-04-01', '2031-04-30')]:
            files = day_files(root, panel, split)
            expected = set(pd.date_range(start, end).strftime('%Y-%m-%d'))
            if set(files) != expected:
                raise ValueError(f'{panel}/{split}: daily files differ from expected calendar')
            targets = template(root, panel, split)
            if set(targets.timestamp.str[:10]) - expected:
                raise ValueError('Template dates outside split calendar')
            record = dict(panel=panel, split=split, daily_files=len(files),
                          target_rows=len(targets),
                          regime_counts={str(k): int(v) for k, v in targets.mask_regime.value_counts().items()},
                          sample_checks=[])
            by_day = targets.groupby(targets.timestamp.str[:10]).mask_regime.agg(['first', 'nunique'])
            if not by_day['nunique'].eq(1).all():
                raise ValueError('A calendar day appears under multiple regimes')
            # Directory sorting groups R1 first. Deliberately cover regimes instead.
            candidates = []
            for regime in ['R1', 'R2', 'R3']:
                dates = sorted(by_day.index[by_day['first'].eq(regime)])
                if dates:
                    candidates.append(dates[0])
            candidates += [d for d in sorted(files) if d not in candidates]
            sampled = candidates[:args.max_days] if args.max_days else candidates
            full_files = day_files(root, panel, 'train', False) if split == 'train' else None
            if full_files and set(full_files) != expected:
                raise ValueError('Unmasked train calendar is incomplete')
            for date in sampled:
                frame = read_day(files[date])
                target = targets.loc[targets.timestamp.str.startswith(date)]
                positions = select_rows(frame, target)
                assert_hidden(frame, positions)
                if set(frame.link_id) - set(net.index):
                    raise ValueError('Observation link missing from network')
                selected = frame.iloc[positions]
                if 'mask_regime' in selected and set(selected.mask_regime) != set(target.mask_regime):
                    raise ValueError('Target/file regime mismatch')
                if full_files:
                    complete = read_day(full_files[date])
                    joined = target[JOIN_KEY].merge(complete, on=JOIN_KEY, how='left', validate='one_to_one')
                    if not eligible(joined).all():
                        raise ValueError('Template includes unavailable/non-finite training truth')
                record['sample_checks'].append(dict(date=date, observation_rows=len(frame),
                      template_targets=len(target),
                      blank_speed_rows=int(frame.speed_kmh.isna().sum()),
                      station_count=int(frame.station_id.nunique()), link_count=int(frame.link_id.nunique()),
                      columns=list(frame.columns)))
            report['panels'].append(record)
            print(f'[{panel}/{split}] calendar OK, {len(targets):,} targets; checked {len(sampled)} days', flush=True)
    if args.output.exists():
        raise FileExistsError(f'Audit output already exists: {args.output}')
    save_json(args.output, report)
    print(f'Audit saved: {args.output}', flush=True)
