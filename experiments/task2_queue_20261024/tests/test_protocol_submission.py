import json
import numpy as np
import pandas as pd
import pytest
from task2q.protocol import role_for_day, split_manifest, write_frozen_split
from task2q.submission import validate_queue


def test_calendar_is_disjoint_complete_and_reserved(tmp_path):
    m = split_manifest()
    assert len(m['date_roles']) == 273
    assert {r: list(m['date_roles'].values()).count(r) for r in ['fit','tune','calibrate','audit']} == {
        'fit':183, 'tune':31, 'calibrate':31, 'audit':28}
    assert role_for_day('2030-11-30') == 'fit'
    assert role_for_day('2030-12-01') == 'tune'
    with pytest.raises(ValueError):
        role_for_day('2031-03-01')
    path = tmp_path/'frozen.json'
    write_frozen_split(path)
    changed = json.loads(path.read_text())
    changed['date_roles']['2031-02-01'] = 'fit'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='differs'):
        write_frozen_split(path)


def frame():
    return pd.DataFrame(dict(window_id=['w','w'],timestamp=['2031-03-01T00:05:00Z','2031-03-01T00:10:00Z'],
                             link_id=['l','l'],queue_pred=[0,1]))


def test_submission_exact_set_binary_and_order_independent():
    f=frame()
    assert validate_queue(f.iloc[::-1],f)['passed']
    for values in ([.5,1],[np.nan,1],[2,0]):
        bad=f.copy();bad['queue_pred']=values
        with pytest.raises(ValueError):validate_queue(bad,f)
    with pytest.raises(ValueError):validate_queue(f.iloc[:1],f)
    with pytest.raises(ValueError):validate_queue(pd.concat([f,f.iloc[:1]]),f)


def test_no_formal_training_action():
    from task2q.cli import arguments
    with pytest.raises(SystemExit):
        arguments(['train','--release-root','.'])
