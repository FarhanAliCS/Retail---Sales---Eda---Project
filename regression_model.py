"""
Regression add-on for the Retail Sales EDA project.
Predicts profit per order from order details.

Run AFTER eda_project.py (it needs data/retail_orders_clean.csv):
    pip install scikit-learn
    python regression_model.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------ CONFIG
CSV_PATH = "data/retail_orders_clean.csv"
TARGET = "profit"
NUMERIC_FEATURES = ["unit_price", "quantity", "discount", "customer_age", "shipping_days"]
CATEGORICAL_FEATURES = ["category", "region", "customer_segment", "payment_method"]
DATE_COLUMN = "order_date"
RANDOM_STATE = 42
# -------------------------------------------------------------------------
# Leakage note: "revenue" is calculated from unit_price, quantity and discount,
# and "profit_margin_pct" is calculated from profit itself, so neither is used
# as an input feature.

df = pd.read_csv(CSV_PATH)

if DATE_COLUMN:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df["month"] = df[DATE_COLUMN].dt.month
    df["weekday"] = df[DATE_COLUMN].dt.weekday
    NUMERIC_FEATURES = NUMERIC_FEATURES + ["month", "weekday"]

df = df.dropna(subset=[TARGET])
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), NUMERIC_FEATURES),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
])

models = {
    "Baseline (predict the mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
}

rows, fitted = [], {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2").mean()
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2 (test)": r2_score(y_test, pred),
        "R2 (5-fold CV)": cv_r2,
    })
    fitted[name] = (pipe, pred)

results = pd.DataFrame(rows).round(3)
print("\n=== RESULTS (lower MAE/RMSE is better, higher R2 is better) ===")
print(results.to_string(index=False))

best = results.sort_values("R2 (test)", ascending=False).iloc[0]["Model"]
pred = fitted[best][1]
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.5)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual " + TARGET)
plt.ylabel("Predicted " + TARGET)
plt.title(f"Actual vs Predicted ({best})")
plt.legend()
plt.tight_layout()
plt.savefig("figures/08_actual_vs_predicted.png", dpi=150)
print(f"\nSaved plot: figures/08_actual_vs_predicted.png  (best model: {best})")

errors = (y_test - pred).abs().sort_values(ascending=False).head(5)
print("\nFive largest errors (index, absolute error):")
print(errors.round(2).to_string())
