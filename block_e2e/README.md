---
title: Zurich Apartment Rent Predictor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# Zurich Apartment Rent Predictor

A machine learning application that predicts monthly rental prices for apartments in the canton of Zurich.

## Dataset

The dataset contains 2344 apartment listings from the canton of Zurich with municipality-level demographic and economic data. After removing missing values and duplicates, the cleaned dataset has 2344 rows and 45 columns.

**Original features:** bfs_number, rooms, area, price, postalcode, address, town, description_raw, bfs_name, pop, pop_dens, frg_pct, emp, tax_income, lat, lon

**Meaning of municipality features:**
- `pop`: population (number of residents)
- `pop_dens`: population density (residents per km2)
- `frg_pct`: percentage of foreign residents
- `emp`: number of employees
- `tax_income`: median taxable income

## Preprocessing Steps

- Removed rows with missing values (`dropna`)
- Removed duplicate rows (`drop_duplicates`)
- Computed new feature `distance_to_zurich_hb`: haversine distance (km) from apartment coordinates to Zurich main station (HB)
- Used existing derived feature `room_per_m2` (= area / rooms, i.e. m2 per room)
- Used binary features `furnished` and `zurich_city` extracted from listing descriptions and location

## New Feature: distance_to_zurich_hb

This feature was not used in any prior exercise. It computes the geographic distance in kilometers from each apartment's coordinates to Zurich Hauptbahnhof (main station) at lat=47.3769, lon=8.5417 using the haversine formula. Proximity to the city center is a strong predictor of rental prices.

## Iterative Modeling Process

### Iteration 1 - Baseline

| | |
|---|---|
| **Objective** | Establish baseline performance using original municipality features |
| **Changes** | Initial iteration, no prior model |
| **Preprocessing** | dropna, drop_duplicates |
| **Features (7)** | rooms, area, pop, pop_dens, frg_pct, emp, tax_income |
| **Validation** | 5-Fold Cross-Validation |

| Model | Hyperparameters | CV Mean R2 | CV Std R2 | CV Mean RMSE | CV Std RMSE |
|-------|----------------|------------|-----------|--------------|-------------|
| Linear Regression | default | 0.5394 | 0.0878 | 678 | 208 |
| Random Forest | n_estimators=100, random_state=42 | 0.5364 | 0.0511 | 674 | 169 |

**Diagnosis:** Both models show moderate performance. Random Forest does not improve over Linear Regression, indicating the baseline features alone have limited predictive power. The high variance in RMSE suggests the models struggle with certain subgroups (e.g. Zurich city apartments).

---

### Iteration 2 - Enhanced Features + Tuned Models

| | |
|---|---|
| **Objective** | Improve performance by adding distance-based and categorical features, and tuning model hyperparameters |
| **Changes** | Added 4 new features: distance_to_zurich_hb (NEW), room_per_m2, furnished, zurich_city. Used 3 models with tuned hyperparameters. |
| **Preprocessing** | Same as iteration 1 + haversine distance computation from lat/lon |
| **Features (11)** | rooms, area, pop, pop_dens, frg_pct, emp, tax_income, distance_to_zurich_hb, room_per_m2, furnished, zurich_city |
| **Validation** | 5-Fold Cross-Validation |

| Model | Hyperparameters | CV Mean R2 | CV Std R2 | CV Mean RMSE | CV Std RMSE |
|-------|----------------|------------|-----------|--------------|-------------|
| Ridge Regression | alpha=1.0 | 0.5378 | 0.0855 | 679 | 206 |
| Random Forest | n_estimators=300, max_depth=20, min_samples_leaf=5, random_state=42 | 0.5986 | 0.0687 | 628 | 173 |
| Gradient Boosting | n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42 | 0.6136 | 0.0503 | 613 | 144 |

**Diagnosis:** Gradient Boosting achieves the best R2 of 0.6136, an improvement of +0.077 over the baseline Random Forest. The additional features (especially distance_to_zurich_hb and zurich_city) help capture location-based price variation. RMSE standard deviation decreased from 169 to 144, indicating more consistent predictions across folds.

---

## Summary Table

| Iteration | Objective | Models | Best CV R2 | Best CV RMSE | Change |
|-----------|-----------|--------|------------|--------------|--------|
| 1 | Baseline with 7 features | Linear Regression, Random Forest | 0.5394 | 674 | - |
| 2 | +4 features, tuned models | Ridge, Random Forest, Gradient Boosting | 0.6136 | 613 | +0.074 R2 |

## Final Selected Model

**Gradient Boosting Regressor** with the following configuration:
- `n_estimators=300`
- `max_depth=5`
- `learning_rate=0.1`
- `random_state=42`

Selected because it achieved the highest cross-validated R2 (0.6136) and lowest RMSE (613) with the most stable performance across folds (lowest std deviation).

## Application

The web application is built with Gradio. Users can:
1. Select a town from 111 municipalities in the canton of Zurich
2. Enter the number of rooms and apartment area in m2
3. Indicate whether the apartment is furnished
4. Get an estimated monthly rent in CHF

Municipality-level data (population, tax income, etc.) and distance to Zurich HB are automatically looked up from the selected town.

## How to Run Locally

```bash
uv add gradio scikit-learn pandas numpy
uv run python app.py
```

## Files

- `app.py` - Gradio web application
- `train.py` - Training script with documented iterations
- `model.pkl` - Trained Gradient Boosting model
- `municipality_lookup.csv` - Municipality reference data for predictions
- `apartments_data.csv` - Training dataset
- `requirements.txt` - Python dependencies for Hugging Face Spaces
