import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"
DEFAULT_HIGH_CUT = 0.60
MAX_ROWS = 5000
CATS = ["code_module", "region", "highest_education"]
NUMS = ["num_of_prev_attempts", "studied_credits", "date_registration",
        "clicks_pre", "clicks_early", "clicks_late", "active_days", "n_sites",
        "days_since_last_active", "n_assessments_due", "n_submitted"]
REQUIRED = CATS + NUMS + ["mean_score"]

st.set_page_config(page_title="Student Early-Warning Tool", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load(MODELS / "risk_model_day28.joblib")
    meta = json.loads((MODELS / "model_meta.json").read_text())
    opts = json.loads((MODELS / "app_options.json").read_text())
    return model, meta, opts


model, meta, opts = load_assets()
d = opts["defaults"]


def predict(df):
    x = df.copy()
    for c in NUMS + ["mean_score"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["total_clicks"] = x["clicks_pre"] + x["clicks_early"] + x["clicks_late"]
    x["click_trend"] = x["clicks_late"] - x["clicks_early"]
    x["no_clicks_flag"] = (x["total_clicks"] == 0).astype(int)
    return model.predict_proba(x[meta["cat_features"] + meta["num_features"]])[:, 1]


def to_band(risk, medium, high):
    return np.where(risk >= high, "High", np.where(risk >= medium, "Medium", "Low"))


st.title("Student Early-Warning Tool")
st.write("Estimates the risk that a student will withdraw or fail a course, "
         "using only what is known after the first 4 weeks.")

with st.sidebar:
    st.header("Risk bands")
    threshold = st.slider("Medium band starts at", 0.10, 0.80,
                          float(meta["threshold"]), 0.05)
    high_cut = st.slider("High band starts at", 0.30, 0.95, DEFAULT_HIGH_CUT, 0.05)
    if high_cut <= threshold:
        high_cut = threshold + 0.05
        st.caption("High cutoff raised to stay above the Medium cutoff.")
    st.caption("Lower cutoffs catch more at-risk students but raise more false alarms. "
               "The defaults were chosen on training data.")

tab1, tab2 = st.tabs(["Single student", "Class upload"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Student profile")
        module = st.selectbox("Course module", opts["code_module"])
        region = st.selectbox("Region", opts["region"])
        education = st.selectbox("Highest education", opts["highest_education"])
        prev = st.number_input("Previous attempts", 0, 10, int(d["num_of_prev_attempts"]))
        credits = st.number_input("Studied credits", 0, 700, int(d["studied_credits"]), step=10)
        reg_day = st.number_input("Registration day (negative = before course start)",
                                  -400, 100, int(d["date_registration"]))
    with col2:
        st.subheader("Activity, first 4 weeks")
        pre = st.number_input("Clicks before course start", 0, 100000, int(d["clicks_pre"]), step=10)
        early = st.number_input("Clicks, days 0-13", 0, 100000, int(d["clicks_early"]), step=10)
        late = st.number_input("Clicks, days 14-28", 0, 100000, int(d["clicks_late"]), step=10)
        active_days = st.number_input("Days with any activity", 0, 60, int(d["active_days"]))
        n_sites = st.number_input("Different course pages visited", 0, 500, int(d["n_sites"]))
        last_gap = st.number_input("Days since last activity (29 if never active)",
                                   0, 60, int(d["days_since_last_active"]))
    with col3:
        st.subheader("Early assessments")
        n_due = st.number_input("Assessments due by day 28", 0, 10, 1)
        if n_due > 0:
            submitted = st.number_input("Submitted by day 28", 0, int(n_due), min(1, int(n_due)))
        else:
            submitted = 0
            st.caption("No assessments are due yet for this course.")
        if submitted > 0:
            score = st.number_input("Average score (0-100)", 0.0, 100.0, 70.0, step=1.0)
        else:
            score = np.nan

    if ((pre + early + late) == 0) != (active_days == 0):
        st.warning("Total clicks and days-with-activity disagree "
                   "(both should be zero together).")

    row = {"code_module": module, "region": region, "highest_education": education,
           "num_of_prev_attempts": prev, "studied_credits": credits,
           "date_registration": reg_day, "clicks_pre": pre, "clicks_early": early,
           "clicks_late": late, "active_days": active_days, "n_sites": n_sites,
           "days_since_last_active": last_gap, "n_assessments_due": n_due,
           "n_submitted": submitted, "mean_score": score}
    risk = float(predict(pd.DataFrame([row]))[0])
    band = str(to_band(np.array([risk]), threshold, high_cut)[0])

    st.divider()
    r1, r2 = st.columns([1, 3])
    r1.metric("Estimated risk", f"{risk:.0%}")
    with r2:
        st.progress(min(max(risk, 0.0), 1.0))
        if band == "High":
            st.error("**High risk.** Prioritise a supportive check-in.")
        elif band == "Medium":
            st.warning("**Medium risk.** Keep this student on a watch list "
                       "and check again next week.")
        else:
            st.success("**Low risk.** No action needed now, but this does not "
                       "guarantee the student will do well.")

with tab2:
    st.warning("This is a public demo. Do not upload real student data. "
               "Use the template or synthetic data only.")
    template = pd.DataFrame([
        {"student_id": "S001", "code_module": "AAA", "region": "East Anglian Region",
         "highest_education": "Lower Than A Level", "num_of_prev_attempts": 1,
         "studied_credits": 60, "date_registration": -10, "clicks_pre": 0,
         "clicks_early": 0, "clicks_late": 0, "active_days": 0, "n_sites": 0,
         "days_since_last_active": 29, "n_assessments_due": 1, "n_submitted": 0,
         "mean_score": None},
        {"student_id": "S002", "code_module": "AAA", "region": "East Anglian Region",
         "highest_education": "HE Qualification", "num_of_prev_attempts": 0,
         "studied_credits": 60, "date_registration": -60, "clicks_pre": 150,
         "clicks_early": 250, "clicks_late": 300, "active_days": 25, "n_sites": 30,
         "days_since_last_active": 0, "n_assessments_due": 1, "n_submitted": 1,
         "mean_score": 85},
        {"student_id": "S003", "code_module": "AAA", "region": "East Anglian Region",
         "highest_education": "A Level or Equivalent", "num_of_prev_attempts": 0,
         "studied_credits": 60, "date_registration": -30, "clicks_pre": 20,
         "clicks_early": 80, "clicks_late": 40, "active_days": 10, "n_sites": 12,
         "days_since_last_active": 8, "n_assessments_due": 1, "n_submitted": 1,
         "mean_score": 55},
    ])
    st.download_button("Download template CSV", template.to_csv(index=False),
                       "template.csv", "text/csv")
    st.caption("Required columns: " + ", ".join(REQUIRED) +
               ". Optional: student_id. Leave mean_score blank if nothing was submitted.")

    up = st.file_uploader("Upload a CSV", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            st.error("Could not read this file as a CSV.")
            st.stop()
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
            st.stop()
        if len(df) > MAX_ROWS:
            st.error(f"Too many rows ({len(df):,}). The limit is {MAX_ROWS:,}.")
            st.stop()
        bad = df[NUMS].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
        if bad.any():
            st.warning(f"{int(bad.sum())} row(s) with missing or non-numeric values "
                       "were skipped.")
            df = df[~bad].copy()
        if df.empty:
            st.error("No valid rows to score.")
            st.stop()
        for c in CATS:
            unknown = set(df[c].astype(str)) - set(opts[c])
            if unknown:
                st.warning(f"Unrecognised {c} values (the model treats them as unknown): "
                           + ", ".join(sorted(unknown)[:5]))

        out = df.copy()
        r = predict(df)
        out["risk_pct"] = (r * 100).round(1)
        out["band"] = to_band(r, threshold, high_cut)
        out = out.sort_values("risk_pct", ascending=False)

        counts = out["band"].value_counts()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Students scored", f"{len(out):,}")
        m2.metric("High", int(counts.get("High", 0)))
        m3.metric("Medium", int(counts.get("Medium", 0)))
        m4.metric("Low", int(counts.get("Low", 0)))
        st.dataframe(out, use_container_width=True)
        st.download_button("Download scored CSV", out.to_csv(index=False),
                           "scored_students.csv", "text/csv")

with st.expander("About this tool and its limits"):
    st.markdown(f"""
- Trained on the Open University Learning Analytics Dataset (UK distance learning, CC BY 4.0), predicting at day 28 of a course. Students who had already withdrawn by day 28 are excluded.
- On held-out students the model's ROC-AUC was **{meta['test_roc_auc']:.2f}**. At the {meta['threshold']:.2f} cutoff it caught about **{meta['test_recall']:.0%}** of at-risk students, and about **{meta['test_precision']:.0%}** of flagged students were truly at risk.
- **Bands on held-out students:** about 78% of the High band, 46% of the Medium band and 21% of the Low band ended up withdrawing or failing (the overall rate was 43%). The High band held 25% of students but 46% of all at-risk students. The Medium band is only a little above the overall rate, so treat it as a watch group.
- The score is a ranking aid, not a calibrated probability.
- A flag should lead to a supportive conversation, never a penalty. A person should review every flag.
- Gender, disability, age band and deprivation band are **not** used by the model. This narrowed the gap in flag rates for disabled students, but gaps by gender and by prior education remained, because other features act as proxies. Region and prior education are still used and were not audited in depth.
- It may not transfer to other institutions or course types.
""")