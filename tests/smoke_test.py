import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "risk_model_day28.joblib")
meta = json.loads((ROOT / "models" / "model_meta.json").read_text())

MEDIUM, HIGH = 0.35, 0.60


def band(risk):
    return "High" if risk >= HIGH else "Medium" if risk >= MEDIUM else "Low"


def score(edu, prev, reg, pre, early, late, active, sites, gap, due, submitted, mean_score):
    total = pre + early + late
    row = {
        "code_module": "AAA", "region": "East Anglian Region",
        "highest_education": edu, "num_of_prev_attempts": prev,
        "studied_credits": 60, "date_registration": reg,
        "total_clicks": total, "active_days": active, "n_sites": sites,
        "clicks_pre": pre, "clicks_early": early, "clicks_late": late,
        "click_trend": late - early, "days_since_last_active": gap,
        "no_clicks_flag": int(total == 0),
        "n_submitted": submitted, "mean_score": mean_score,
        "n_assessments_due": due,
    }
    X = pd.DataFrame([row])[meta["cat_features"] + meta["num_features"]]
    return float(model.predict_proba(X)[0, 1])


a = score("Lower Than A Level", 1, -10, 0, 0, 0, 0, 0, 29, 1, 0, np.nan)
b = score("HE Qualification", 0, -60, 150, 250, 300, 25, 30, 0, 1, 1, 85.0)

print(f"Student A (disengaged): {a:.0%} -> {band(a)}")
print(f"Student B (engaged):    {b:.0%} -> {band(b)}")
print("PASS" if band(a) == "High" and band(b) == "Low" else "CHECK THIS")