"""Uses tiny synthetic CSVs with the same columns as the real sample/ files. No transcript text."""
import numpy as np
import pandas as pd
import pytest

from src import scoring as sc
from src import predictions_table as pt

STEPS = """config,task_id,episode_id,split,quarter,step_index,message_index,eligible_rank,n_total,n_eligible,relative_position,n_tool_calls,target_tool_name,has_visible_text
cfgA,1,1__honest,pilot,early,3,6,2,20,10,0.15,1,bash,true
cfgA,1,1__honest,pilot,late,18,36,9,20,10,0.9,2,bash,false
cfgB,1,1__honest,pilot,early,4,8,2,20,10,0.2,1,text_editor,false
cfgB,1,1__honest,pilot,late,17,34,9,20,10,0.85,1,bash,false
"""
EPS = """config,task_id,split,episode_id,main_task_score,main_task_success,attempt,run
cfgA,1,pilot,1__honest,0.5,I,1,r
cfgB,1,pilot,1__honest,1.0,C,2,r
"""
LEAK = """config,task_id,split,quarter,step_index,extraction_status,n_targets,target_kinds,command_leak_reasoning,command_leak_visible,command_leak,target_leak_reasoning,target_leak_visible,target_leak,basename_only_mention,target_in_earlier_reasoning,command_core_leak,leak_category
cfgA,1,pilot,early,3,ok,1,path,false,false,false,true,false,true,false,false,false,target_only
cfgA,1,pilot,late,18,ambiguous,1,path,false,false,false,false,false,false,true,true,false,none
cfgB,1,pilot,early,4,ok,1,path,true,false,true,true,false,true,false,false,false,command
cfgB,1,pilot,late,17,no_target,0,,false,false,false,false,false,false,false,false,false,none
"""


@pytest.fixture
def sample_dir(tmp_path):
    for name, text in (("steps.csv", STEPS), ("episodes.csv", EPS), ("leak_flags.csv", LEAK)):
        (tmp_path / name).write_text(text)
    return str(tmp_path)


def preds(step_ids, rng):
    rows = []
    for s in step_ids:
        for c in ("T0", "T", "R-only", "T+R"):
            rows.append(dict(step_id=s, condition=c, type_match=bool(rng.random() < .8), target_match=bool(rng.random() < .5),
                             parse_fail=False))
    return pd.DataFrame(rows)


def test_step_table_columns_and_flags(sample_dir):
    t = pt.load_step_table(sample_dir)
    assert len(t) == 4 and t.step_id.is_unique
    assert t.loc[t.step_id == "cfgA|1|18", "multi_call"].item() is True or t.loc[t.step_id == "cfgA|1|18", "multi_call"].item()
    assert set(t.position) == {1, 4}
    assert t.loc[t.model == "cfgB", "attempt_gt1"].all() and not t.loc[t.model == "cfgA", "attempt_gt1"].any()
    assert t.loc[t.step_id == "cfgB|1|4", "verbatim_leak"].item()
    assert t.loc[t.step_id == "cfgA|1|3", "leak_target"].item()
    assert t.loc[t.step_id == "cfgA|1|18", "ambiguous_extraction"].item()
    assert t.episode_id.nunique() == 2  # same episode_id in two configs stays distinct


def test_without_leak_file(sample_dir):
    import os
    os.remove(os.path.join(sample_dir, "leak_flags.csv"))
    t = pt.load_step_table(sample_dir)
    assert "leak_target" not in t.columns and len(t) == 4


def test_main_split_leak_file_is_loaded(sample_dir):
    """scripts/leak_flags.py --split main writes leak_flags_main.csv; its flags must not be silently dropped."""
    import os
    path = os.path.join(sample_dir, "leak_flags.csv")
    header, *rows = open(path).read().splitlines()
    pilot = [r for r in rows if not r.startswith("cfgB,")]
    main = [r.replace(",pilot,", ",main,") for r in rows if r.startswith("cfgB,")]
    open(path, "w").write("\n".join([header] + pilot) + "\n")
    open(os.path.join(sample_dir, "leak_flags_main.csv"), "w").write("\n".join([header] + main) + "\n")
    t = pt.load_step_table(sample_dir)
    assert t.has_leak_flags.all()
    assert t.loc[t.step_id == "cfgB|1|4", "verbatim_leak"].item()
    assert t.loc[t.step_id == "cfgA|1|3", "leak_target"].item()

def test_to_scoring_frame_and_scoring_runs(sample_dir):
    t = pt.load_step_table(sample_dir)
    rng = np.random.default_rng(0)
    p = preds(t.step_id, rng)
    p.loc[0, "parse_fail"] = True
    p.loc[0, ["type_match", "target_match"]] = True  # a parse failure must still be scored wrong
    df = pt.to_scoring_frame(t, p)
    assert not df.loc[(df.step_id == p.step_id[0]) & (df.condition == "T0"), "correct"].item()
    assert not df.loc[(df.step_id == p.step_id[0]) & (df.condition == "T0"), "type_correct"].item()
    assert set(df.condition) == set(sc.CONDS)
    sc.check_input(df)
    sub = sc.subset(df, drop=("leak_target",))
    assert "cfgA|1|3" not in set(sub.step_id)
    sc.check_input(sub)


def test_bad_inputs(sample_dir):
    t = pt.load_step_table(sample_dir)
    p = preds(t.step_id, np.random.default_rng(1))
    with pytest.raises(ValueError):
        pt.to_scoring_frame(t, p.assign(step_id="nope"))
    with pytest.raises(ValueError):
        pt.to_scoring_frame(t, p.assign(condition="X"))
