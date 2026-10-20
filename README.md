# SleepWise

A sleep diary that learns your patterns. Log each night in about 30 seconds. SleepWise predicts how
well you'll sleep, explains the prediction with SHAP, and becomes personal as your diary grows.

**Pages:** Home · Log sleep · Trends · Tonight · Insights · Profile. There's also a demo account
with six weeks of simulated data, useful for presentations.

---

## How the ML works

```
prediction = general model (trained on the Kaggle dataset)  +  your personal correction
```

| Part | Algorithm | Trained on | Why |
|---|---|---|---|
| General model | Random Forest classifier → expected quality on a 1–10 scale | Sleep Health & Lifestyle dataset | Gives sensible predictions from the first day |
| Personal model | Ridge regression on the general model's errors | **Your own diary** | Learns how *you* differ from the average person; also uses caffeine, screen time and bedtime, which the dataset doesn't have |
| Explanations | TreeSHAP (forest) + linear SHAP (ridge) | – | Shows exactly how much each habit moved the prediction; the contributions add up to the prediction |

- The algorithm never changes. Each user gets their own ridge model, re-fitted from their diary whenever they open the app.
- It switches on after **7 nights**. Ridge regularisation keeps the correction small while there's little data, so early predictions stay close to the general model.
- After 10 nights, the app measures cross-validated error for both models and shows it, for example "±0.6 points vs ±0.7 for the general model". This is your evidence that personalisation helps.
- `train.py` still runs the six-model comparison from Review 2 for your report. It isn't shown to app users.

---

## 1. Prerequisites

| You need | Notes |
|---|---|
| Python **3.11 or 3.12** | SHAP depends on `numba`, which can lag behind the newest Python versions |
| Git + a **GitHub** account | Streamlit Cloud deploys from a GitHub repo |
| A **Supabase** account (free) | Hosted PostgreSQL database for user accounts and diaries |
| A **Streamlit Community Cloud** account (free) | Sign in with GitHub at share.streamlit.io |
| The real Kaggle CSV (optional but recommended) | *Sleep Health and Lifestyle Dataset*. Check its licence on Kaggle before committing it to a public repo |

## 2. Run locally

```bash
cd sleepwise
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

With no configuration, data is stored in `data/sleepwise.db` (SQLite). Click **Explore with a demo account** to try it.

**Use the real dataset:** save it as `data/Sleep_health_and_lifestyle_dataset.csv` and delete `models/population.joblib`.
The model retrains automatically on the next start.

**Tests:** `pip install pytest && pytest -q`

## 3. Create the database (Supabase)

On Streamlit Cloud, local files can be wiped when the app restarts, so diaries must live in a hosted database.

1. supabase.com → **New project**. Pick a region near your users (e.g. Mumbai) and **save the database password**.
2. Open the project and click **Connect** at the top. Copy the **Session pooler** connection string
   (it looks like `postgresql://postgres.abcd…:[YOUR-PASSWORD]@aws-0-ap-south-1.pooler.supabase.com:5432/postgres`).
   Use the pooler rather than the direct connection, because the direct one may need IPv6, which Streamlit Cloud may not support.
3. Replace `[YOUR-PASSWORD]` with your password. If the password contains symbols like `@ # / ?`, URL-encode them (e.g. `@` → `%40`).
4. Test it locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, paste the URL, run
   `streamlit run app.py` and create an account. SleepWise creates its tables (`sw_users`, `sw_entries`) automatically.
5. **Lock down the tables.** Supabase also exposes tables through a public REST API. In **SQL Editor**, run:
   ```sql
   alter table sw_users   enable row level security;
   alter table sw_entries enable row level security;
   ```
   With no policies, the public API can't read them. The app still can, because it connects as the database owner.

## 4. Push to GitHub

```bash
git init
git add .
git commit -m "SleepWise: diary, personal model, SHAP"
git branch -M main
git remote add origin https://github.com/<your-username>/sleepwise.git
git push -u origin main
```

`.gitignore` keeps `secrets.toml`, the local database and trained models out of the repo. **Never commit your database password.**
The included GitHub Actions workflow runs the tests on every push.

## 5. Deploy (Streamlit Community Cloud)

1. share.streamlit.io → **Create app** → **Deploy a public app from GitHub**.
2. Repository `your-username/sleepwise`, branch `main`, main file `app.py`. Choose a custom URL such as `sleepwise-mepco`.
3. **Advanced settings** → Python **3.12** → **Secrets**: paste
   ```toml
   DATABASE_URL = "postgresql://postgres.…:…@…pooler.supabase.com:5432/postgres"
   ```
4. **Deploy.** The first build takes a few minutes, mostly installing SHAP. Every `git push` redeploys automatically.

Free apps sleep after a period of no visitors. Open the link a few minutes before a review so it's awake.

## 6. Project layout

```
app.py                      Streamlit app: login, navigation, all pages
sleepwise/store.py          database (SQLite or PostgreSQL via SQLAlchemy), bcrypt password hashing
sleepwise/population.py     general model + TreeSHAP
sleepwise/personal.py       personal ridge model, combined SHAP, what-if, diary patterns
sleepwise/demo.py           demo account with simulated diary
sleepwise/data.py           dataset cleaning (duplicates, BMI labels, BP split, labels)
sleepwise/model.py, train.py  six-model comparison for the project report
sleepwise/ui.py             theme and components
tests/test_core.py          unit tests (auth, upsert, cold start, SHAP additivity, ...)
```

## Notes and limitations

- Occupation was removed from the model inputs. The dataset's job list doesn't cover students, the app's main users.
- The dataset only covers ages 27–59, sleep of 5.8–8.5 h and stress of 3–8. The app warns users who fall outside these ranges,
  so students under 27 will always see the age warning.
- Login here is simple username and password with bcrypt hashing. It's fine for a student project, but it has no password reset or email verification.
- Associations are not causation. SleepWise is not a medical tool.
