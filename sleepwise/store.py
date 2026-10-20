"""Persistence: users, profiles and diary entries.

Uses SQLAlchemy, so the same code runs on
  - SQLite (local file, default) and
  - PostgreSQL (e.g. Supabase) when DATABASE_URL is set in .streamlit/secrets.toml or the environment.
"""
from __future__ import annotations

import os
from datetime import date, datetime, time

import bcrypt
import pandas as pd
from sqlalchemy import (Column, Date, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, Text, UniqueConstraint, create_engine, delete, insert, select, update)
from sqlalchemy.engine import Engine

from .data import ROOT

meta = MetaData()

users = Table(
    "sw_users", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(40), unique=True, nullable=False),
    Column("display_name", String(60), nullable=False),
    Column("password_hash", String(100), nullable=False),
    Column("created_at", DateTime, nullable=False),
    # profile
    Column("gender", String(10)),
    Column("age", Integer),
    Column("bmi_category", String(20)),
    Column("heart_rate", Float),
    Column("systolic_bp", Float),
    Column("diastolic_bp", Float),
    Column("sleep_goal", Float, default=8.0),
)

entries = Table(
    "sw_entries", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("sw_users.id", ondelete="CASCADE"), nullable=False),
    Column("night", Date, nullable=False),          # the morning you woke up
    Column("bedtime", String(5), nullable=False),   # "23:30"
    Column("waketime", String(5), nullable=False),  # "07:00"
    Column("sleep_hours", Float, nullable=False),
    Column("quality", Integer, nullable=False),     # 1-10, self-rated
    Column("stress", Integer, nullable=False),      # 1-10, previous day
    Column("activity_min", Integer, nullable=False),
    Column("steps", Integer, nullable=False),
    Column("caffeine", Integer, nullable=False),    # cups after noon
    Column("screen_min", Integer, nullable=False),  # screen time in the hour before bed
    Column("notes", Text),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("user_id", "night", name="uq_user_night"),
)

ENTRY_FIELDS = ["night", "bedtime", "waketime", "sleep_hours", "quality", "stress",
                "activity_min", "steps", "caffeine", "screen_min", "notes"]
PROFILE_FIELDS = ["gender", "age", "bmi_category", "heart_rate", "systolic_bp", "diastolic_bp", "sleep_goal"]


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
        except Exception:
            url = None
    if url and url.startswith("postgres://"):  # SQLAlchemy wants postgresql://
        url = "postgresql://" + url[len("postgres://"):]
    return url or f"sqlite:///{ROOT / 'data' / 'sleepwise.db'}"


def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    eng = create_engine(url, **kwargs)
    meta.create_all(eng)
    return eng


# ------------------------------------------------------------------ helpers

def sleep_hours(bed: time, wake: time) -> float:
    b = bed.hour * 60 + bed.minute
    w = wake.hour * 60 + wake.minute
    mins = (w - b) % (24 * 60)
    return round(mins / 60, 2)


def bedtime_offset(hhmm: str) -> float:
    """Hours relative to midnight: 23:30 -> -0.5, 01:15 -> 1.25. Handles evening/after-midnight."""
    h, m = map(int, hhmm.split(":"))
    x = h + m / 60
    return x - 24 if x >= 12 else x


# ------------------------------------------------------------------ users / auth

class AuthError(Exception):
    pass


def create_user(eng: Engine, username: str, display_name: str, password: str) -> int:
    username = username.strip().lower()
    if len(username) < 3 or not username.replace("_", "").replace(".", "").isalnum():
        raise AuthError("Username must be at least 3 characters: letters, numbers, . or _")
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters.")
    pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with eng.begin() as c:
        if c.execute(select(users.c.id).where(users.c.username == username)).first():
            raise AuthError("That username is taken.")
        res = c.execute(insert(users).values(
            username=username, display_name=display_name.strip() or username,
            password_hash=pw, created_at=datetime.utcnow(), sleep_goal=8.0))
        return int(res.inserted_primary_key[0])


def authenticate(eng: Engine, username: str, password: str) -> int:
    with eng.connect() as c:
        row = c.execute(select(users.c.id, users.c.password_hash)
                        .where(users.c.username == username.strip().lower())).first()
    if not row or not bcrypt.checkpw(password.encode(), row.password_hash.encode()):
        raise AuthError("Incorrect username or password.")
    return int(row.id)


def get_user(eng: Engine, user_id: int) -> dict:
    with eng.connect() as c:
        row = c.execute(select(users).where(users.c.id == user_id)).mappings().first()
    return dict(row) if row else {}


def update_profile(eng: Engine, user_id: int, **fields) -> None:
    fields = {k: v for k, v in fields.items() if k in PROFILE_FIELDS + ["display_name"]}
    with eng.begin() as c:
        c.execute(update(users).where(users.c.id == user_id).values(**fields))


def delete_user(eng: Engine, user_id: int) -> None:
    with eng.begin() as c:
        c.execute(delete(entries).where(entries.c.user_id == user_id))
        c.execute(delete(users).where(users.c.id == user_id))


def profile_complete(user: dict) -> bool:
    return all(user.get(k) is not None for k in ["gender", "age", "bmi_category"])


# ------------------------------------------------------------------ diary

def save_entry(eng: Engine, user_id: int, **e) -> None:
    """Insert or replace the entry for that night."""
    values = {k: e[k] for k in ENTRY_FIELDS if k in e}
    values.update(user_id=user_id, created_at=datetime.utcnow())
    with eng.begin() as c:
        c.execute(delete(entries).where((entries.c.user_id == user_id) & (entries.c.night == values["night"])))
        c.execute(insert(entries).values(**values))


def save_entries_bulk(eng: Engine, user_id: int, rows: list[dict]) -> None:
    now = datetime.utcnow()
    with eng.begin() as c:
        c.execute(delete(entries).where(entries.c.user_id == user_id))
        c.execute(insert(entries), [{**{k: r.get(k) for k in ENTRY_FIELDS}, "user_id": user_id,
                                     "created_at": now} for r in rows])


def delete_entry(eng: Engine, user_id: int, night: date) -> None:
    with eng.begin() as c:
        c.execute(delete(entries).where((entries.c.user_id == user_id) & (entries.c.night == night)))


def load_entries(eng: Engine, user_id: int) -> pd.DataFrame:
    with eng.connect() as c:
        rows = c.execute(select(*[entries.c[k] for k in ENTRY_FIELDS])
                         .where(entries.c.user_id == user_id).order_by(entries.c.night)).mappings().all()
    df = pd.DataFrame(rows, columns=ENTRY_FIELDS)
    df["night"] = pd.to_datetime(df["night"])
    df["bed_offset"] = df["bedtime"].map(bedtime_offset).astype(float)
    return df
