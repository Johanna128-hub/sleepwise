"""SleepWise — personal sleep diary with ML predictions.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sleepwise import demo, store
from sleepwise.personal import FULL_ENTRIES, MIN_ENTRIES, PERSONAL_FEATURES, PersonalModel, patterns, quality_label
from sleepwise.population import PopulationModel
from sleepwise.ui import (ACCENT, ACCENT_2, BORDER, CLASS_COLORS, GOOD, MOON, MUTED, POOR, SURFACE, TEXT,
                          chart, header, inject_css, kpi, pill, quality_color, section, style)

st.set_page_config(page_title="SleepWise", page_icon="🌙", layout="wide", initial_sidebar_state="collapsed")
inject_css()
st.logo("assets/logo.svg", size="large", icon_image="assets/logo.svg")


# ------------------------------------------------------------------ shared resources
@st.cache_resource(show_spinner="Preparing SleepWise…")
def engine():
    return store.get_engine()


@st.cache_resource(show_spinner="Loading the sleep model…")
def population_model() -> PopulationModel:
    return PopulationModel.load_or_train()


ENG = engine()
POP = population_model()


def current_user() -> dict | None:
    uid = st.session_state.get("user_id")
    if uid is None:
        return None
    u = store.get_user(ENG, uid)
    if not u:
        st.session_state.pop("user_id", None)
    return u or None


def diary(uid: int) -> pd.DataFrame:
    return store.load_entries(ENG, uid)


def personal_model(user: dict, d: pd.DataFrame) -> PersonalModel:
    return PersonalModel.fit(POP, user, d)


def fmt_h(x: float) -> str:
    h, m = int(x), int(round((x % 1) * 60))
    if m == 60:
        h, m = h + 1, 0
    return f"{h}h {m:02d}m"


def fmt_clock(offset: float) -> str:
    h = (offset + 24) % 24
    mins = int(round(h * 60)) % 1440
    return f"{mins // 60:02d}:{mins % 60:02d}"


def streak(d: pd.DataFrame) -> int:
    if d.empty:
        return 0
    days = set(d["night"].dt.date)
    cur = date.today() if date.today() in days else date.today() - timedelta(days=1)
    n = 0
    while cur in days:
        n += 1
        cur -= timedelta(days=1)
    return n


# ================================================================== auth page
def auth_page():
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"""<div class="sw-hero">
<div class="sw-brand">{MOON.format(s=34)}<span>SleepWise</span></div>
<h1>Understand your sleep.<br>One morning at a time.</h1>
<p>Log last night in 30 seconds. SleepWise learns which of your habits go with your best nights,
and shows what to change tonight.</p>
<div class="sw-feat"><div class="dot">1</div><div><b>Sleep diary</b><div class="s">Bedtime, wake time, how rested you feel, and yesterday's habits.</div></div></div>
<div class="sw-feat"><div class="dot">2</div><div><b>Learns your patterns</b><div class="s">After a week, predictions adapt to you, not just the average person.</div></div></div>
<div class="sw-feat"><div class="dot">3</div><div><b>Explains itself</b><div class="s">See exactly which habits pushed tonight's prediction up or down.</div></div></div>
</div>""", unsafe_allow_html=True)
    with right:
        st.write("")
        with st.container(border=True):
            t1, t2 = st.tabs(["Log in", "Create account"])
            with t1:
                with st.form("login", border=False):
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    if st.form_submit_button("Log in", type="primary", use_container_width=True):
                        try:
                            st.session_state.user_id = store.authenticate(ENG, u, p)
                            st.rerun()
                        except store.AuthError as e:
                            st.error(str(e))
            with t2:
                with st.form("signup", border=False):
                    n = st.text_input("Your name")
                    u = st.text_input("Choose a username")
                    p = st.text_input("Password", type="password", help="At least 6 characters")
                    p2 = st.text_input("Confirm password", type="password")
                    if st.form_submit_button("Create account", type="primary", use_container_width=True):
                        if p != p2:
                            st.error("Passwords don't match.")
                        else:
                            try:
                                st.session_state.user_id = store.create_user(ENG, u, n, p)
                                st.rerun()
                            except store.AuthError as e:
                                st.error(str(e))
            st.markdown('<div class="sw-muted" style="text-align:center;margin:.4rem 0">or</div>',
                        unsafe_allow_html=True)
            if st.button("Explore with a demo account", use_container_width=True):
                st.session_state.user_id = demo.reset_demo(ENG)
                st.rerun()
        st.markdown('<div class="sw-muted" style="margin-top:.8rem">SleepWise is a student project for '
                    'self-reflection, not a medical device.</div>', unsafe_allow_html=True)


# ================================================================== onboarding / profile form
BMI_HELP = "BMI = weight (kg) ÷ height (m)². Under 25 → Normal, 25–30 → Overweight, 30+ → Obese."


def profile_form(user: dict, key: str, submit_label: str) -> bool:
    with st.form(key):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", user.get("display_name") or "")
        genders = ["Female", "Male", "Prefer not to say"]
        g = c2.selectbox("Gender", genders, index=genders.index(user["gender"]) if user.get("gender") in genders else 0)
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 13, 100, int(user.get("age") or 20))
        bmis = ["Normal", "Overweight", "Obese"]
        bmi = c2.selectbox("BMI category", bmis, help=BMI_HELP,
                           index=bmis.index(user["bmi_category"]) if user.get("bmi_category") in bmis else 0)
        goal = st.slider("Nightly sleep goal (hours)", 6.0, 10.0, float(user.get("sleep_goal") or 8.0), 0.25)
        with st.expander("Optional: resting heart rate and blood pressure"):
            known = st.checkbox("I know these values", value=user.get("heart_rate") is not None)
            c1, c2, c3 = st.columns(3)
            hr = c1.number_input("Resting HR (bpm)", 40, 120, int(user.get("heart_rate") or 70))
            sbp = c2.number_input("Systolic", 90, 180, int(user.get("systolic_bp") or 120))
            dbp = c3.number_input("Diastolic", 50, 120, int(user.get("diastolic_bp") or 80))
            st.caption("If you leave these out, SleepWise uses typical values.")
        if st.form_submit_button(submit_label, type="primary"):
            store.update_profile(ENG, user["id"], display_name=name or user["username"], gender=g, age=int(age),
                                 bmi_category=bmi, sleep_goal=float(goal),
                                 heart_rate=float(hr) if known else None,
                                 systolic_bp=float(sbp) if known else None,
                                 diastolic_bp=float(dbp) if known else None)
            return True
    return False


def onboarding_page():
    user = current_user()
    header("Welcome", f"Hi {user['display_name']}, let's set up your profile",
           "A few basics help SleepWise compare you with people like you until it has learned your own patterns.")
    if profile_form(user, "onboard", "Save and continue"):
        st.rerun()


# ================================================================== home
def home_page():
    user = current_user()
    d = diary(user["id"])
    goal = float(user.get("sleep_goal") or 8)
    header(datetime.now().strftime("%A, %d %B"), f"Hi {user['display_name']}")

    logged_today = (not d.empty) and (d["night"].dt.date == date.today()).any()
    if not logged_today:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f'<div style="font-weight:600;font-size:1.05rem;color:{TEXT}">How did you sleep last night?</div>'
                        f'<div class="sw-muted">Logging takes about 30 seconds.</div>', unsafe_allow_html=True)
            if c2.button("Log last night", type="primary", use_container_width=True):
                st.switch_page(PAGES["log"])

    if d.empty:
        st.info("Your statistics will appear here after your first entry.")
        return

    last = d.iloc[-1]
    wk = d[d["night"] > pd.Timestamp(date.today() - timedelta(days=7))]
    c = st.columns(4)
    kpi(c[0], "Last night", f"{last['quality']}<small>/10</small>",
        f"{fmt_h(last['sleep_hours'])} · {last['night']:%a %d %b}")
    if len(wk):
        kpi(c[1], "Avg sleep · 7 days", fmt_h(wk["sleep_hours"].mean()),
            f"{wk['sleep_hours'].mean() - goal:+.1f} h vs your {goal:g} h goal")
        kpi(c[2], "Avg quality · 7 days", f"{wk['quality'].mean():.1f}<small>/10</small>", f"{len(wk)} night{'s' if len(wk) != 1 else ''} logged")
    else:
        kpi(c[1], "Avg sleep · 7 days", "–", "no entries this week")
        kpi(c[2], "Avg quality · 7 days", "–", "")
    s = streak(d)
    kpi(c[3], "Logging streak", f"{s}<small> day{'s' if s != 1 else ''}</small>", "keep it going" if s else "log today to start")

    section("Last 14 nights")
    r = d[d["night"] > pd.Timestamp(date.today() - timedelta(days=14))]
    f = go.Figure(go.Bar(x=r["night"], y=r["sleep_hours"], marker_color=[quality_color(q) for q in r["quality"]],
                         customdata=np.stack([r["quality"], r["bedtime"], r["waketime"]], axis=1),
                         hovertemplate="%{x|%a %d %b}<br>%{y:.2f} h · quality %{customdata[0]}/10"
                                       "<br>%{customdata[1]} → %{customdata[2]}<extra></extra>"))
    f.add_hline(y=goal, line=dict(color=TEXT, dash="dot", width=1), annotation_text=f"goal {goal:g} h",
                annotation_font_color=MUTED)
    f.update_yaxes(title="Hours", rangemode="tozero")
    f.update_xaxes(tickformat="%d %b", range=[pd.Timestamp(date.today() - timedelta(days=14)) + pd.Timedelta(hours=12),
                                              pd.Timestamp(date.today()) + pd.Timedelta(hours=12)])
    chart(style(f, 280, False))
    st.markdown(f'<div class="sw-muted">Bar colour = how you rated the night: '
                f'<span style="color:{GOOD}">■</span> good (8–10) · <span style="color:#F5C26B">■</span> fair (7) · '
                f'<span style="color:{POOR}">■</span> poor (≤6)</div>', unsafe_allow_html=True)

    a, b = st.columns(2, gap="medium")
    with a:
        section("SleepWise is learning you")
        learning_card(len(d), personal_model(user, d))
    with b:
        section("Your strongest pattern")
        ps = patterns(d, goal)
        if ps:
            insight_card(ps[0])
            if st.button("See all insights →"):
                st.switch_page(PAGES["insights"])
        else:
            st.markdown('<div class="sw-card sw-muted">Patterns appear after about a week of entries.</div>',
                        unsafe_allow_html=True)


def learning_card(n: int, pm: PersonalModel):
    pct = min(1.0, n / FULL_ENTRIES)
    if n < MIN_ENTRIES:
        status = f"{MIN_ENTRIES - n} more night{'s' if MIN_ENTRIES - n != 1 else ''} until predictions become personal."
    elif pm.mae_personal is not None and pm.mae_population is not None:
        status = (f"Personal model active. On nights it hasn't seen, it predicts your rating within "
                  f"<b>±{pm.mae_personal:.1f}</b> points on average (general model: ±{pm.mae_population:.1f}).")
    else:
        status = "Personal model active. Accuracy is shown after 10 nights."
    st.markdown(f'<div class="sw-card"><div class="sw-kpi-label">{n} night{"s" if n != 1 else ""} logged</div>'
                f'<div class="sw-progress"><div style="width:{pct*100:.0f}%"></div></div>'
                f'<div class="sw-muted">{status}</div></div>', unsafe_allow_html=True)


def insight_card(p: dict):
    better = p["diff"] >= 0
    col = GOOD if better else POOR
    txt = (f"On nights when {p['good']}, you rated your sleep "
           f"<b>{abs(p['diff']):.1f} points {'higher' if better else 'lower'}</b>.")
    st.markdown(f'<div class="sw-insight"><div class="h"><div class="t">{p["habit"]}</div>'
                f'<div class="d" style="color:{col}">{p["diff"]:+.1f}</div></div>'
                f'<div class="b">{txt} Average {p["mean_a"]:.1f} over {p["n_a"]} nights vs '
                f'{p["mean_b"]:.1f} over {p["n_b"]} nights.</div></div>', unsafe_allow_html=True)


# ================================================================== log sleep
def log_page():
    user = current_user()
    d = diary(user["id"])
    header("Sleep diary", "Log a night", "Fill this in after you wake up. Habits refer to the day before.")

    night = st.date_input("Morning of", date.today(), max_value=date.today(), format="DD/MM/YYYY")
    existing = d[d["night"].dt.date == night]
    e = existing.iloc[0].to_dict() if len(existing) else {}
    if e:
        st.markdown('<div class="sw-note">You already logged this night. Saving will update it.</div>',
                    unsafe_allow_html=True)

    def t(s, default):
        return datetime.strptime(s, "%H:%M").time() if s else default

    with st.form("entry"):
        section("Sleep", top=True)
        c1, c2 = st.columns(2)
        bed = c1.time_input("Went to bed", t(e.get("bedtime"), time(23, 0)), step=timedelta(minutes=5))
        wake = c2.time_input("Woke up", t(e.get("waketime"), time(7, 0)), step=timedelta(minutes=5))
        quality = st.select_slider("How well did you sleep?", options=list(range(1, 11)),
                                   value=int(e.get("quality", 7)),
                                   help="1 = terrible, 10 = perfectly rested")
        section("Yesterday")
        c1, c2 = st.columns(2)
        stress = c1.slider("Stress level", 1, 10, int(e.get("stress", 5)), help="1 = calm, 10 = very stressed")
        screen = c2.slider("Screen time in the hour before bed (min)", 0, 60 * 2, int(e.get("screen_min", 30)), 5)
        c1, c2, c3 = st.columns(3)
        activity = c1.number_input("Exercise / active minutes", 0, 300, int(e.get("activity_min", 30)), 5)
        steps = c2.number_input("Steps", 0, 40000, int(e.get("steps", 6000)), 500)
        caffeine = c3.number_input("Caffeinated drinks after noon", 0, 10, int(e.get("caffeine", 0)))
        notes = st.text_input("Notes (optional)", e.get("notes") or "", placeholder="e.g. exam tomorrow, late dinner")
        hours = store.sleep_hours(bed, wake)
        st.caption(f"Time in bed: {fmt_h(hours)}")
        saved = st.form_submit_button("Update entry" if e else "Save entry", type="primary")

    if saved:
        if hours < 1 or hours > 16:
            st.error("Bedtime and wake time give an unusual sleep length. Please check them.")
            return
        entry = dict(night=night, bedtime=bed.strftime("%H:%M"), waketime=wake.strftime("%H:%M"),
                     sleep_hours=hours, quality=int(quality), stress=int(stress), activity_min=int(activity),
                     steps=int(steps), caffeine=int(caffeine), screen_min=int(screen), notes=notes or None)
        # What would SleepWise have predicted, using only earlier nights?
        prior = d[d["night"].dt.date < night]
        pm = personal_model(user, prior)
        pred = pm.predict({**entry, "bed_offset": store.bedtime_offset(entry["bedtime"])}).final
        store.save_entry(ENG, user["id"], **entry)
        st.success(f"Saved. You rated it {quality}/10. From your habits, SleepWise expected about {pred:.1f}.")

    if not d.empty:
        section("Recent entries")
        r = d.sort_values("night", ascending=False).head(14)
        show = pd.DataFrame({
            "Night": r["night"].dt.strftime("%a %d %b"), "Bed": r["bedtime"], "Wake": r["waketime"],
            "Hours": r["sleep_hours"].round(2), "Quality": r["quality"], "Stress": r["stress"],
            "Steps": r["steps"], "Caffeine": r["caffeine"], "Screen (min)": r["screen_min"],
        })
        st.dataframe(show, hide_index=True, use_container_width=True)
        with st.expander("Delete an entry"):
            pick = st.selectbox("Night", r["night"].dt.date, format_func=lambda x: x.strftime("%a %d %b %Y"))
            if st.button("Delete this entry"):
                store.delete_entry(ENG, user["id"], pick)
                st.rerun()


# ================================================================== trends (sleep statistics)
def trends_page():
    user = current_user()
    d = diary(user["id"])
    goal = float(user.get("sleep_goal") or 8)
    header("Sleep statistics", "Your trends")
    if d.empty:
        st.info("Log a few nights to see your trends.")
        return
    span = st.segmented_control("Period", ["7 days", "30 days", "90 days", "All"], default="30 days",
                                label_visibility="collapsed") or "30 days"
    if span != "All":
        d = d[d["night"] > pd.Timestamp(date.today() - timedelta(days=int(span.split()[0])))]
    if d.empty:
        st.info("No entries in this period.")
        return

    debt = (goal - d["sleep_hours"]).clip(lower=0).sum()
    c = st.columns(4)
    kpi(c[0], "Average sleep", fmt_h(d["sleep_hours"].mean()), f"goal {goal:g} h")
    kpi(c[1], "Average quality", f"{d['quality'].mean():.1f}<small>/10</small>",
        f"{(d['quality'] >= 8).mean():.0%} of nights rated 8+")
    kpi(c[2], "Sleep debt", fmt_h(debt), f"hours short of goal over {len(d)} nights")
    sd = d["bed_offset"].std() * 60 if len(d) > 1 else 0
    kpi(c[3], "Bedtime consistency", f"±{sd:.0f}<small> min</small>",
        f"typical bedtime {fmt_clock(d['bed_offset'].median())}")
    st.write("")

    a, b = st.columns(2, gap="large")
    with a:
        f = go.Figure()
        f.add_trace(go.Scatter(x=d["night"], y=d["quality"], mode="markers", name="Nightly rating",
                               marker=dict(color=[quality_color(q) for q in d["quality"]], size=8)))
        if len(d) >= 3:
            f.add_trace(go.Scatter(x=d["night"], y=d["quality"].rolling(7, min_periods=3).mean(), mode="lines",
                                   name="7-night average", line=dict(color=ACCENT, width=3, shape="spline")))
        f.update_yaxes(range=[0.5, 10.5], title="Quality")
        f.update_layout(title="Sleep quality")
        chart(style(f, 320))
    with b:
        f = go.Figure(go.Bar(x=d["night"], y=d["sleep_hours"], base=d["bed_offset"],
                             marker_color=[quality_color(q) for q in d["quality"]], marker_line_width=0,
                             customdata=np.stack([d["bedtime"], d["waketime"], d["sleep_hours"]], axis=1),
                             hovertemplate="%{x|%a %d %b}<br>%{customdata[0]} → %{customdata[1]}"
                                           " (%{customdata[2]:.2f} h)<extra></extra>"))
        lo = np.floor(d["bed_offset"].min()) - 1
        hi = np.ceil((d["bed_offset"] + d["sleep_hours"]).max()) + 1
        ticks = np.arange(lo, hi + 1, 2)
        f.update_yaxes(tickvals=ticks, ticktext=[fmt_clock(t) for t in ticks], range=[hi, lo])
        f.update_layout(title="Sleep window (bedtime → wake time)", bargap=.35)
        chart(style(f, 320, False))

    a, b = st.columns(2, gap="large")
    with a:
        f = go.Figure(go.Bar(x=d["night"], y=d["sleep_hours"], marker_color=ACCENT_2, opacity=.85,
                             hovertemplate="%{x|%a %d %b}: %{y:.2f} h<extra></extra>"))
        f.add_hline(y=goal, line=dict(color=TEXT, dash="dot", width=1), annotation_text="goal",
                    annotation_font_color=MUTED)
        f.update_layout(title="Hours slept")
        f.update_yaxes(rangemode="tozero")
        chart(style(f, 300, False))
    with b:
        order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        w = d.assign(day=d["night"].dt.strftime("%a")).groupby("day").agg(q=("quality", "mean"),
                                                                           h=("sleep_hours", "mean"), n=("quality", "size"))
        w = w.reindex([x for x in order if x in w.index])
        f = go.Figure(go.Bar(x=w.index, y=w["q"], marker_color=[quality_color(q) for q in w["q"]],
                             customdata=np.stack([w["h"], w["n"]], axis=1),
                             hovertemplate="%{x}: quality %{y:.1f}<br>%{customdata[0]:.1f} h · %{customdata[1]} nights<extra></extra>"))
        f.update_layout(title="By day of the week (morning)")
        f.update_yaxes(range=[0, 10])
        chart(style(f, 300, False))


# ================================================================== tonight (prediction + SHAP)
def tonight_page():
    user = current_user()
    d = diary(user["id"])
    pm = personal_model(user, d)
    header("Plan tonight", "How will you sleep tonight?",
           "Tell SleepWise about today and your plan. It predicts your sleep quality and explains why.")
    recent = d.tail(7)
    dflt = (lambda col, v: float(recent[col].mean()) if len(recent) else v)

    left, right = st.columns([5, 7], gap="large")
    with left:
        with st.container(border=True):
            section("Tonight's plan", top=True)
            bed = st.time_input("Planned bedtime", time(23, 0) if d.empty else
                                datetime.strptime(fmt_clock(round(recent["bed_offset"].median() * 12) / 12), "%H:%M").time(),
                                step=timedelta(minutes=15))
            hrs = st.slider("Planned hours of sleep", 4.0, 10.0, round(dflt("sleep_hours", 7.5) * 4) / 4, 0.25)
            section("Today so far")
            stress = st.slider("Stress", 1, 10, int(round(dflt("stress", 5))))
            screen = st.slider("Screen time in the hour before bed (min)", 0, 120, int(dflt("screen_min", 30) // 5 * 5), 5)
            c1, c2 = st.columns(2)
            act = c1.number_input("Active minutes", 0, 300, int(dflt("activity_min", 30) // 5 * 5), 5)
            steps = c2.number_input("Steps", 0, 40000, int(dflt("steps", 6000) // 100 * 100), 500)
            caf = st.number_input("Caffeinated drinks after noon", 0, 10, 0)

    night = {"sleep_hours": hrs, "stress": stress, "activity_min": act, "steps": steps, "caffeine": caf,
             "screen_min": screen, "bed_offset": store.bedtime_offset(bed.strftime("%H:%M"))}
    pred = pm.predict(night)

    with right:
        c1, c2 = st.columns([1, 1.2], vertical_alignment="center")
        with c1:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=pred.final, number=dict(valueformat=".1f", font=dict(size=44, color=TEXT)),
                gauge=dict(axis=dict(range=[1, 10], tickcolor=MUTED, tickfont=dict(color=MUTED)),
                           bar=dict(color=CLASS_COLORS[pred.label], thickness=.28), bgcolor=SURFACE, borderwidth=0,
                           steps=[dict(range=[1, 6.5], color="rgba(242,119,122,.12)"),
                                  dict(range=[6.5, 7.5], color="rgba(245,194,107,.12)"),
                                  dict(range=[7.5, 10], color="rgba(111,211,163,.12)")])))
            chart(style(g, 210, False).update_layout(margin=dict(t=25, b=0, l=30, r=30)))
        with c2:
            st.markdown(f'<div>{pill(pred.label + " night expected")}</div>'
                        f'<div class="sw-muted" style="margin-top:.6rem">Predicted sleep quality out of 10.</div>'
                        f'<div class="sw-muted" style="margin-top:.4rem">'
                        + ("Personalised from your diary." if pred.personalised else
                           f"Based on people like you. It becomes personal after {MIN_ENTRIES} logged nights "
                           f"(you have {len(d)}).") + '</div>', unsafe_allow_html=True)
        oor = POP.out_of_range(user, night)
        if oor:
            st.markdown(f'<div class="sw-note">Your {", ".join(oor)} is outside the range of the reference data, '
                        'so treat this prediction with extra caution.</div>', unsafe_allow_html=True)

        section("Why this prediction")
        base, contrib = pm.explain(pd.DataFrame([night]))
        waterfall(base, contrib.iloc[0], pred.final)
        st.caption("Each bar shows how much a factor moved the prediction from the starting point "
                   "(a typical night). Calculated with SHAP.")

    section("What would help most tonight?")
    wi = pm.what_if(night)
    rows = ""
    for _, r in wi.iterrows():
        col = GOOD if r["Change"] > 0.05 else (POOR if r["Change"] < -0.05 else MUTED)
        rows += (f'<div class="sw-lever"><div><div class="name">{r["Habit"]}</div>'
                 f'<div class="detail">{r["From"]} → {r["To"]}</div></div>'
                 f'<div class="delta" style="color:{col}">{r["Change"]:+.1f}</div></div>')
    st.markdown(f'<div class="sw-card">{rows}</div>', unsafe_allow_html=True)
    st.caption("Change in predicted quality (out of 10) if you made just that one change. These are patterns "
               "in data, not guarantees or medical advice.")


def waterfall(base: float, contrib: pd.Series, final: float, top: int = 6):
    c = contrib[contrib.abs() > 1e-6].sort_values(key=abs, ascending=False)
    main, rest = c.head(top), c.iloc[top:].sum()
    labels = ["Starting point"] + list(main.index) + (["Other factors"] if abs(rest) > 0.01 else []) + ["Prediction"]
    vals = [base] + list(main.values) + ([rest] if abs(rest) > 0.01 else []) + [0]
    measures = ["absolute"] + ["relative"] * (len(vals) - 2) + ["total"]
    text = [f"{base:.1f}"] + [f"{v:+.2f}" for v in vals[1:-1]] + [f"{final:.1f}"]
    f = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=vals, measure=measures, text=text, textposition="outside",
        increasing=dict(marker=dict(color=GOOD)), decreasing=dict(marker=dict(color=POOR)),
        totals=dict(marker=dict(color=ACCENT)), connector=dict(line=dict(color=BORDER)),
        hoverinfo="skip"))
    f.data[0].base = None
    lo = min(base, final, base + np.cumsum(vals[1:-1]).min() if len(vals) > 2 else base) - 1
    hi = max(base, final, base + np.cumsum(vals[1:-1]).max() if len(vals) > 2 else base) + 1
    f.update_xaxes(range=[max(0, lo), min(11, hi)])
    f.update_yaxes(autorange="reversed")
    chart(style(f, 60 + 38 * len(labels), False))


# ================================================================== insights
def insights_page():
    user = current_user()
    d = diary(user["id"])
    goal = float(user.get("sleep_goal") or 8)
    header("Insights", "What goes with your best nights",
           "Comparisons from your own diary, plus what the personal model has learned.")
    if len(d) < 6:
        st.info(f"Insights appear after 6 nights. You've logged {len(d)}.")
        return
    pm = personal_model(user, d)
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        section("Your patterns", top=True)
        ps = patterns(d, goal)
        for p in ps:
            insight_card(p)
        if not ps:
            st.markdown('<div class="sw-card sw-muted">Not enough variety in your entries yet.</div>',
                        unsafe_allow_html=True)
    with b:
        section("What moves your sleep the most", top=True)
        imp = pm.importance(d).head(8).sort_values()
        f = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                             marker_color=[ACCENT_2 if i in ("Sleep duration", "Stress", "Physical activity", "Steps",
                                                             "Caffeine after noon", "Screen time before bed", "Bedtime")
                                           else "#5B6599" for i in imp.index],
                             hovertemplate="%{y}: ±%{x:.2f} points on average<extra></extra>"))
        f.update_xaxes(title="Average effect on predicted quality (points)")
        chart(style(f, 330, False))
        st.caption("Average size of each factor's SHAP value across your logged nights. "
                   "Teal = habits you can change.")
        section("How well SleepWise knows you")
        learning_card(len(d), pm)
    st.markdown('<div class="sw-muted" style="margin-top:1.2rem">These are associations in your own data, '
                'not proof of cause and effect. If poor sleep persists, talk to a doctor.</div>',
                unsafe_allow_html=True)


# ================================================================== profile
def profile_page():
    user = current_user()
    header("Account", "Profile & data")
    a, b = st.columns([3, 2], gap="large")
    with a:
        if profile_form(user, "profile", "Save profile"):
            st.toast("Profile saved")
            st.rerun()
    with b:
        with st.container(border=True):
            section("Your data", top=True)
            d = diary(user["id"])
            st.markdown(f'<div class="sw-muted">{len(d)} diary entries · signed in as <b>{user["username"]}</b></div>',
                        unsafe_allow_html=True)
            st.write("")
            st.download_button("Download my diary (CSV)",
                               d.drop(columns=["bed_offset"], errors="ignore").to_csv(index=False).encode(),
                               "sleepwise_diary.csv", "text/csv", use_container_width=True)
            if user["username"] == demo.DEMO_USER:
                if st.button("Reset demo data", use_container_width=True):
                    demo.reset_demo(ENG)
                    st.rerun()
            if st.button("Log out", use_container_width=True):
                st.session_state.clear()
                st.rerun()
            with st.expander("Delete account"):
                st.caption("Permanently deletes your profile and every diary entry.")
                ok = st.checkbox("I understand this can't be undone")
                if st.button("Delete my account", disabled=not ok, type="primary"):
                    store.delete_user(ENG, user["id"])
                    st.session_state.clear()
                    st.rerun()


# ================================================================== navigation
user = current_user()
if user is None:
    nav = st.navigation([st.Page(auth_page, title="Welcome", icon=":material/bedtime:")], position="hidden")
elif not store.profile_complete(user):
    nav = st.navigation([st.Page(onboarding_page, title="Set up", icon=":material/person:")], position="hidden")
else:
    PAGES = {
        "home": st.Page(home_page, title="Home", icon=":material/home:", default=True),
        "log": st.Page(log_page, title="Log sleep", icon=":material/edit_note:", url_path="log"),
        "trends": st.Page(trends_page, title="Trends", icon=":material/monitoring:", url_path="trends"),
        "tonight": st.Page(tonight_page, title="Tonight", icon=":material/bedtime:", url_path="tonight"),
        "insights": st.Page(insights_page, title="Insights", icon=":material/lightbulb:", url_path="insights"),
        "profile": st.Page(profile_page, title="Profile", icon=":material/account_circle:", url_path="profile"),
    }
    nav = st.navigation(list(PAGES.values()), position="top")
nav.run()
