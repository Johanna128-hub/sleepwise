"""Demo account with six weeks of simulated diary entries (for presentations)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from . import store

DEMO_USER, DEMO_PASS = "demo", "sleepwise-demo"


def simulated_diary(days: int = 42, seed: int = 7, end: date | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    end = end or date.today()
    rows = []
    for i in range(days):
        night = end - timedelta(days=days - 1 - i)
        weekend = night.weekday() in (5, 6)          # Sat/Sun mornings follow Fri/Sat nights
        bed = float(np.clip(rng.normal(0.3 if weekend else -0.1, 0.7), -2, 2.5))  # hours from midnight
        screen = int(np.clip(rng.normal(35, 25), 0, 150) // 5 * 5)
        caffeine = int(rng.choice([0, 0, 1, 1, 2, 3], p=[.3, .2, .2, .15, .1, .05]))
        stress = int(np.clip(round(rng.normal(4.5 if weekend else 5.5, 1.6)), 1, 10))
        activity = int(np.clip(rng.normal(55 if weekend else 35, 18), 0, 150) // 5 * 5)
        steps = int(np.clip(2500 + 70 * activity + rng.normal(0, 1200), 1500, 16000) // 100 * 100)
        sleep = float(np.clip(rng.normal(7.3 if weekend else 7.0, 0.65) - 0.15 * caffeine - 0.3 * bed, 4.5, 10))
        sleep = round(sleep * 4) / 4
        wake = bed + sleep
        q = (7.0 + 0.9 * (sleep - 7) - 0.25 * (stress - 5) - 0.025 * (screen - 30)
             - 0.45 * caffeine - 0.5 * bed + 0.00008 * (steps - 6000) + rng.normal(0, 0.5))
        rows.append({
            "night": night, "bedtime": _hhmm(bed), "waketime": _hhmm(wake), "sleep_hours": sleep,
            "quality": int(np.clip(round(q), 1, 10)), "stress": stress, "activity_min": activity,
            "steps": steps, "caffeine": caffeine, "screen_min": screen, "notes": None,
        })
    return rows


def _hhmm(offset: float) -> str:
    h = (offset + 24) % 24
    mins = int(round(h * 60)) % (24 * 60)
    return f"{mins // 60:02d}:{mins % 60:02d}"


def reset_demo(eng) -> int:
    """(Re)create the demo user with fresh data ending today. Returns user id."""
    try:
        uid = store.authenticate(eng, DEMO_USER, DEMO_PASS)
    except store.AuthError:
        uid = store.create_user(eng, DEMO_USER, "Demo", DEMO_PASS)
    store.update_profile(eng, uid, display_name="Demo", gender="Female", age=21, bmi_category="Normal",
                         heart_rate=None, systolic_bp=None, diastolic_bp=None, sleep_goal=8.0)
    store.save_entries_bulk(eng, uid, simulated_diary())
    return uid
