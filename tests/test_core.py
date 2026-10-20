"""Run with:  pytest -q"""
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from sleepwise import demo, store
from sleepwise.data import FEATURES, load_dataset
from sleepwise.personal import MIN_ENTRIES, PERSONAL_FEATURES, PersonalModel, patterns
from sleepwise.population import PopulationModel


@pytest.fixture(scope="module")
def pop():
    return PopulationModel.train()


@pytest.fixture()
def eng(tmp_path):
    return store.get_engine(f"sqlite:///{tmp_path / 'test.db'}")


def test_no_leaky_columns():
    assert "Sleep Disorder" not in FEATURES and "Person ID" not in FEATURES


def test_duplicates_removed():
    df, rep = load_dataset()
    assert not df.drop(columns=["Person ID"]).duplicated().any()


def test_sleep_hours_across_midnight():
    assert store.sleep_hours(time(23, 30), time(7, 0)) == 7.5
    assert store.sleep_hours(time(1, 0), time(9, 15)) == 8.25
    assert store.bedtime_offset("23:30") == -0.5 and store.bedtime_offset("01:15") == 1.25


def test_auth(eng):
    uid = store.create_user(eng, "alice", "Alice", "secret1")
    assert store.authenticate(eng, "ALICE", "secret1") == uid
    with pytest.raises(store.AuthError):
        store.authenticate(eng, "alice", "wrong")
    with pytest.raises(store.AuthError):
        store.create_user(eng, "alice", "Again", "secret1")


def test_entry_upsert(eng):
    uid = store.create_user(eng, "bob", "Bob", "secret1")
    e = dict(night=date(2026, 10, 1), bedtime="23:00", waketime="07:00", sleep_hours=8.0, quality=7,
             stress=4, activity_min=30, steps=6000, caffeine=0, screen_min=20, notes=None)
    store.save_entry(eng, uid, **e)
    store.save_entry(eng, uid, **{**e, "quality": 9})
    d = store.load_entries(eng, uid)
    assert len(d) == 1 and d["quality"].iloc[0] == 9


def test_population_predictions_in_range(pop):
    rows = pd.DataFrame(demo.simulated_diary(10))
    q = pop.expected_quality({"gender": "Female", "age": 30, "bmi_category": "Normal"}, rows)
    assert q.shape == (10,) and np.all((q >= 1) & (q <= 10))


def test_cold_start_uses_general_model(pop):
    d = pd.DataFrame(demo.simulated_diary(MIN_ENTRIES - 1))
    d["bed_offset"] = d["bedtime"].map(store.bedtime_offset)
    pm = PersonalModel.fit(pop, {"gender": "Male", "age": 30, "bmi_category": "Normal"}, d)
    assert not pm.personalised and pm.predict(d.iloc[0][PERSONAL_FEATURES].to_dict()).correction == 0


def test_shap_is_additive(pop):
    d = pd.DataFrame(demo.simulated_diary(30))
    d["bed_offset"] = d["bedtime"].map(store.bedtime_offset)
    user = {"gender": "Prefer not to say", "age": 22, "bmi_category": "Normal"}
    pm = PersonalModel.fit(pop, user, d)
    base, c = pm.explain(d)
    raw = pop.expected_quality(user, d) + pm.ridge.predict(d[PERSONAL_FEATURES].astype(float))
    assert np.allclose(base + c.sum(axis=1).values, raw, atol=1e-6)


def test_patterns_need_enough_data():
    d = pd.DataFrame(demo.simulated_diary(4))
    d["bed_offset"] = d["bedtime"].map(store.bedtime_offset)
    assert patterns(d, 8.0) == []
