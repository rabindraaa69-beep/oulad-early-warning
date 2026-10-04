# Student Early-Warning Tool (OULAD)

An end-to-end machine learning project that predicts, after the **first 4 weeks** of a course, which students are at risk of **withdrawing or failing**. It includes data cleaning, leakage-safe feature engineering, model comparison, threshold selection, a fairness check, and a Streamlit app.

**Live demo:** oulad-early-warning.streamlit.app

## Key results

Final model: gradient boosting on 18 features, evaluated once on held-out students (5,454 rows, 2,365 at-risk). No student appears in both train and test.

|Metric (threshold 0.35)|Value|
|-|-|
|ROC-AUC|0.782|
|PR-AUC|0.752|
|Recall (at-risk students caught)|0.782|
|Precision (flagged students truly at risk)|0.608|
|F1|0.684|
|Share of students flagged|55.8%|

At the 0.35 threshold the model caught 1,850 of 2,365 at-risk students, missed 515, and raised 1,192 false alarms. The threshold was chosen by 5-fold cross-validation on the training data only, so the test set stayed untouched.

### What predicts risk: behavior beats profile

Ablation at day 28 (same test split, gradient boosting):

|Features used|ROC-AUC|
|-|-|
|Profile only|0.662|
|Profile + clicks|0.747|
|Profile + early assessments|0.756|
|All features|0.786|

What students do in the first weeks (clicks and early assessments) predicts risk much better than who they are. Clicks and assessments each add a similar amount, and they help most together.

### How early can we warn?

!\[How early can we warn](reports/cutoff\_comparison.png)

|Week|ROC-AUC|Recall if top 30% flagged|Recall if top 50% flagged|
|-|-|-|-|
|2|0.737|0.470|0.697|
|4|0.786|0.521|0.734|
|6|0.801|0.551|0.751|
|8|0.825|0.571|0.785|

Even at week 2 the model is useful, and accuracy improves steadily with more data. Later cutoffs remove more early withdrawers, so the later rows are slightly harder than they look. These cutoff and ablation experiments were run with the profile fields (gender, disability, age band, deprivation band) included.

### Model comparison (day 28, 5,576 test rows)

|Model|ROC-AUC|
|-|-|
|Majority-class dummy|0.500|
|Logistic regression, profile only|0.660|
|Logistic regression, profile + behavior|0.784|
|Random forest|0.788|
|Gradient boosting|0.793|

Tree models beat logistic regression only slightly, which means most of the signal sits in a few simple features. Gradient boosting was kept for the app.

## Problem setup

* **Unit of prediction:** one student in one course presentation.
* **Target:** at-risk = Withdrawn or Fail, versus Pass or Distinction.
* **Prediction time:** day 28 (end of week 4). Every feature uses only information from day 28 or earlier.
* **Exclusions:** 5,055 registrations that had already unregistered by day 28 were removed, since the model would otherwise "predict" something that had already happened. That leaves 27,538 rows, 44.1% at risk.
* **Leakage controls:** final\_result and 4 October 2026\_unregistration` are used only to build the target and the filter, never as features. The train/test split is by student ID, because 28,785 unique students account for 32,593 registrations.

## Features

* **Profile:** course module, region, highest education, previous attempts, studied credits, registration day.
* **Activity (up to day 28):** total clicks, active days, pages visited, clicks before the course start, clicks on days 0-13 and 14-28, click trend, days since last activity, a no-activity flag.
* **Early assessments (due and submitted by day 28):** number due, number submitted, mean score. A missing score is kept distinct from a score of zero.

## Fairness check

An early version used gender, disability, age band and deprivation band. Per-group results at the 0.35 threshold on the test set:

|Group|Recall|False-alarm rate|Flagged|
|-|-|-|-|
|Women / Men|0.754 / 0.809|0.388 / 0.387|54% / 57%|
|Not disabled / Disabled|0.771 / 0.873|0.376 / 0.527|54% / 72%|
|Most deprived (IMD 0-10%) / Least deprived (90-100%)|0.903 / 0.701|0.582 / 0.254|75% / 39%|
|Lower than A Level / HE qualification|0.882 / 0.670|0.559 / 0.248|73% / 39%|

I then removed the four profile fields and retrained, which cost only 0.004 ROC-AUC (0.786 to 0.782) and kept the same threshold.

|Group|Recall before, after|False-alarm rate before, after|
|-|-|-|
|Disabled|0.873, 0.810|0.527, 0.456|
|Not disabled|0.771, 0.779|0.376, 0.383|
|Women|0.754, 0.747|0.388, 0.403|
|Men|0.809, 0.812|0.387, 0.376|
|Lower than A Level|0.882, 0.880|0.559, 0.574|
|HE qualification|0.670, 0.674|0.248, 0.253|

**Findings**

* Removing the fields narrowed the disability gap at almost no accuracy cost, so the deployed model does not use them.
* It did **not** remove the gender gap (at-risk women are caught about 6 points less often) or the education gap. Other features act as proxies.
* Groups with higher base rates of risk get both higher recall and more false alarms. A single threshold treats groups differently.
* This is one test split without confidence intervals, groups overlap, and these are patterns in predictions, not proof of cause. The model is **not** claimed to be fair.

## Limitations

* Trained on one UK distance-learning institution. It may not transfer to other courses or institutions.
* The score is a ranking aid, not a calibrated probability.
* The 0.35 threshold flags 56% of students. A school with limited staff would choose a higher threshold or use risk bands.
* A flag should start a supportive conversation, never a penalty, and a person should review every flag.
* Region and prior education are still used and were not audited in depth.
* The model compares students against historical patterns and can't tell why someone is disengaged.

## Run it yourself

```
git clone https://github.com/rabindraaa69-beep/oulad-early-warning.git
cd oulad-early-warning
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m streamlit run app\\app.py
```

To rerun the analysis, install `requirements-dev.txt`, download the OULAD files into `data/raw/`, and run the notebooks in `notebooks/` in order (01 to 05). The final model, threshold and fairness comparison are produced in notebook 05.

```
app/            Streamlit app
models/         saved model, metadata, dropdown options
notebooks/      01 explore, 02 target and cutoff, 03 features, 04 baselines, 05 cutoffs, fairness, final model
reports/        result tables and charts
```

## Data and License

This project uses the Open University Learning Analytics Dataset (OULAD).

* **Source:** https://analyse.kmi.open.ac.uk/open-dataset (downloaded on 4 October 2026)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
* **Citation:** Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171.
* **Changes made:** the original tables were cleaned, filtered, joined and aggregated into per-student features for modeling. No endorsement by the Open University or the dataset authors is implied.
* **Raw data:** not included in this repository because of its size.

The code in this repository is released under the MIT License.

