import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)  # creates the data folder for you
rng = np.random.default_rng(42)
N = 5000

catalog = {
    "Technology": [("Laptop", 900), ("Phone", 600), ("Headphones", 120), ("Monitor", 280)],
    "Furniture": [("Office Chair", 220), ("Desk", 350), ("Bookshelf", 150)],
    "Office Supplies": [("Notebook", 6), ("Printer Paper", 25), ("Pens (Box)", 12), ("Stapler", 15)],
}
regions = ["North", "South", "East", "West"]
region_w = [0.22, 0.20, 0.28, 0.30]
segments = ["Consumer", "Corporate", "Home Office"]

cats = rng.choice(list(catalog), N, p=[0.30, 0.25, 0.45])
prod, price = [], []
for c in cats:
    p, base = catalog[c][rng.integers(len(catalog[c]))]
    prod.append(p)
    price.append(round(base * rng.normal(1, 0.06), 2))

dates = pd.Series(pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, N), unit="D"))
holiday = rng.random(N) < 0.15
dates[holiday] = pd.to_datetime(rng.choice(
    pd.date_range("2023-11-01", "2023-12-31").append(pd.date_range("2024-11-01", "2024-12-31")),
    holiday.sum()))

qty = rng.poisson(2.2, N).clip(1, 12)
disc = rng.choice([0, 0.05, 0.10, 0.20, 0.30], N, p=[.45, .2, .17, .12, .06])
age = rng.normal(38, 12, N).clip(18, 80).round()
region = rng.choice(regions, N, p=region_w)
ship = rng.integers(1, 6, N) + np.where(region == "South", 2, 0)

revenue = np.array(price) * qty * (1 - disc)
margin = {"Technology": 0.18, "Furniture": 0.12, "Office Supplies": 0.30}
profit = revenue * np.array([margin[c] for c in cats]) - disc * np.array(price) * qty * 0.9 + rng.normal(0, 15, N)
rating = (4.3 - 0.12 * (ship - 3) - 2.0 * disc + rng.normal(0, 0.6, N)).clip(1, 5).round(1)

df = pd.DataFrame({
    "order_id": [f"ORD-{i:05d}" for i in range(1, N + 1)],
    "order_date": dates.dt.strftime("%Y-%m-%d"),
    "region": region, "category": cats, "product": prod,
    "unit_price": price, "quantity": qty, "discount": disc,
    "customer_age": age, "customer_segment": rng.choice(segments, N, p=[.5, .3, .2]),
    "payment_method": rng.choice(["Credit Card", "Debit Card", "PayPal", "Cash"], N, p=[.45, .25, .2, .1]),
    "shipping_days": ship, "customer_rating": rating, "profit": profit.round(2),
})

# add real-world mess: missing values, bad values, duplicates
for col, frac in [("customer_age", .06), ("customer_rating", .08), ("shipping_days", .02), ("payment_method", .03)]:
    df.loc[rng.random(N) < frac, col] = np.nan
df.loc[rng.choice(N, 40, replace=False), "region"] = df["region"].str.upper()
df.loc[rng.choice(N, 40, replace=False), "customer_segment"] = " Consumer "
df.loc[rng.choice(N, 8, replace=False), "customer_age"] = rng.choice([150, 999, -5], 8)
df.loc[rng.choice(N, 5, replace=False), "quantity"] = 60
df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/retail_orders_raw.csv", index=False)
print("Done! Saved", df.shape, "to data/retail_orders_raw.csv")