"""
Exploratory Data Analysis & Data Visualization - Retail Orders
Tools: Python | Pandas | NumPy | Matplotlib | Seaborn
Run:   python eda_project.py            (reads data/retail_orders_raw.csv)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/retail_orders_raw.csv")
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 13})


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================== 1. LOAD
df = pd.read_csv(DATA)
print("=" * 60, "\n1. DATA LOADING\n" + "=" * 60)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.info()
print("\nMissing values:\n", df.isna().sum()[lambda s: s > 0])
print("Duplicate rows:", df.duplicated().sum())

# =============================================================== 2. CLEAN
print("\n" + "=" * 60 + "\n2. DATA CLEANING\n" + "=" * 60)
log = {"rows_raw": len(df)}

# 2.1 duplicates
df = df.drop_duplicates()
log["duplicates_removed"] = log["rows_raw"] - len(df)

# 2.2 text standardisation
for col in ["region", "category", "product", "customer_segment", "payment_method"]:
    df[col] = df[col].str.strip()
df["region"] = df["region"].str.title()

# 2.3 types
df["order_date"] = pd.to_datetime(df["order_date"])

# 2.4 impossible values -> NaN (ages must be 18-100)
bad_age = ~df["customer_age"].between(18, 100) & df["customer_age"].notna()
log["invalid_ages_set_to_nan"] = int(bad_age.sum())
df.loc[bad_age, "customer_age"] = np.nan

# 2.5 missing values: numeric -> median (robust to outliers); categorical -> mode
for col in ["customer_age", "shipping_days", "customer_rating"]:
    log[f"missing_{col}"] = int(df[col].isna().sum())
    df[col] = df[col].fillna(df[col].median())
log["missing_payment_method"] = int(df["payment_method"].isna().sum())
df["payment_method"] = df["payment_method"].fillna(df["payment_method"].mode()[0])
df["shipping_days"] = df["shipping_days"].astype(int)

# 2.6 outliers: flag with IQR rule (kept, not deleted - they are real bulk orders)
q1, q3 = df["quantity"].quantile([.25, .75])
df["is_bulk_outlier"] = df["quantity"] > q3 + 1.5 * (q3 - q1)
log["quantity_outliers_flagged"] = int(df["is_bulk_outlier"].sum())
log["rows_clean"] = len(df)
for k, v in log.items():
    print(f"  {k:30s} {v}")
assert df.isna().sum().sum() == 0

# =============================================================== 3. TRANSFORM
df["revenue"] = df["unit_price"] * df["quantity"] * (1 - df["discount"])
df["profit_margin_pct"] = df["profit"] / df["revenue"] * 100
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.to_period("M")
df["month_name"] = df["order_date"].dt.month_name()
df["age_group"] = pd.cut(df["customer_age"], [17, 25, 35, 45, 55, 100],
                         labels=["18-25", "26-35", "36-45", "46-55", "56+"])
df["discount_band"] = pd.cut(df["discount"], [-0.01, 0, 0.10, 1],
                             labels=["No discount", "Low (5-10%)", "High (20-30%)"])
df.to_csv("data/retail_orders_clean.csv", index=False)

# =============================================================== 4. DESCRIPTIVE STATS
print("\n" + "=" * 60 + "\n3. DESCRIPTIVE STATISTICS\n" + "=" * 60)
num_cols = ["unit_price", "quantity", "discount", "customer_age", "shipping_days",
            "customer_rating", "revenue", "profit"]
desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew()
print(desc.round(2))

by_cat = df.groupby("category").agg(orders=("order_id", "count"), revenue=("revenue", "sum"),
                                    profit=("profit", "sum"), avg_rating=("customer_rating", "mean"))
by_cat["margin_%"] = by_cat["profit"] / by_cat["revenue"] * 100
by_region = df.groupby("region").agg(revenue=("revenue", "sum"), profit=("profit", "sum"),
                                     avg_ship=("shipping_days", "mean"), avg_rating=("customer_rating", "mean"))
by_disc = df.groupby("discount_band", observed=True).agg(avg_profit=("profit", "mean"),
                                                         avg_rating=("customer_rating", "mean"))
print("\nBy category:\n", by_cat.round(2))
print("\nBy region:\n", by_region.round(2))
print("\nBy discount band:\n", by_disc.round(2))

# =============================================================== 5. VISUALISATIONS
# 5.1 BAR - revenue & profit by category
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(by_cat)); w = 0.38
b1 = ax.bar(x - w / 2, by_cat["revenue"] / 1e3, w, label="Revenue", color=PALETTE[0])
b2 = ax.bar(x + w / 2, by_cat["profit"] / 1e3, w, label="Profit", color=PALETTE[2])
ax.bar_label(b1, fmt="%.0fK", padding=3); ax.bar_label(b2, fmt="%.0fK", padding=3)
ax.set_xticks(x, by_cat.index); ax.set_ylabel("USD (thousands)")
ax.set_title("Revenue vs Profit by Category"); ax.legend()
save("01_bar_revenue_profit_by_category")

# 5.2 LINE - monthly revenue trend by year
monthly = df.groupby(["year", df["order_date"].dt.month])["revenue"].sum().unstack(0) / 1e3
fig, ax = plt.subplots(figsize=(9, 5))
for yr in monthly.columns:
    ax.plot(monthly.index, monthly[yr], marker="o", lw=2, label=str(yr))
ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_ylabel("Revenue (USD thousands)"); ax.set_title("Monthly Revenue Trend - Year over Year"); ax.legend(title="Year")
pk_month, pk_year = monthly.stack().idxmax()
pk_val = monthly.stack().max()
ax.annotate("Holiday peak", xy=(pk_month, pk_val), xytext=(pk_month - 2.6, pk_val - 12),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5), fontweight="bold")
ax.set_ylim(top=pk_val * 1.08)
save("02_line_monthly_revenue")

# 5.3 HISTOGRAMS - age & revenue (log-ish skew shown)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.histplot(df["customer_age"], bins=25, kde=True, color=PALETTE[0], ax=axes[0])
axes[0].axvline(df["customer_age"].mean(), color="red", ls="--", label=f"Mean {df['customer_age'].mean():.1f}")
axes[0].set_title("Customer Age Distribution"); axes[0].legend()
sns.histplot(df["revenue"], bins=40, kde=True, color=PALETTE[3], ax=axes[1])
axes[1].axvline(df["revenue"].median(), color="black", ls="--", label=f"Median ${df['revenue'].median():,.0f}")
axes[1].set_title("Order Revenue Distribution (right-skewed)"); axes[1].legend()
save("03_histograms_age_revenue")

# 5.4 PIE - category share + payment method
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
rev_cat = df.groupby("category")["revenue"].sum()
axes[0].pie(rev_cat, labels=rev_cat.index, autopct="%1.1f%%", startangle=90, colors=PALETTE,
            wedgeprops=dict(edgecolor="white"))
axes[0].set_title("Revenue Share by Category")
pay = df["payment_method"].value_counts()
axes[1].pie(pay, labels=pay.index, autopct="%1.1f%%", startangle=90, colors=PALETTE[3:],
            wedgeprops=dict(width=0.45, edgecolor="white"))
axes[1].set_title("Payment Methods (orders)")
save("04_pie_category_payment")

# 5.5 SCATTER - discount vs profit, and shipping vs rating
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
samp = df.sample(1500, random_state=1)
sns.scatterplot(data=samp, x="revenue", y="profit", hue="category", alpha=.6, ax=axes[0])
axes[0].set_title("Revenue vs Profit by Category")
sns.regplot(data=df.groupby("shipping_days")["customer_rating"].mean().reset_index(),
            x="shipping_days", y="customer_rating", ax=axes[1], color=PALETTE[1])
axes[1].set_title("Slower Shipping -> Lower Ratings"); axes[1].set_ylabel("Avg rating")
save("05_scatter_relationships")

# 5.6 BOX - profit by discount band, rating by region
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="discount_band", y="profit", hue="discount_band", palette="Set2", legend=False,
            showfliers=False, ax=axes[0])
axes[0].set_title("Profit per Order by Discount Level"); axes[0].set_xlabel("")
sns.boxplot(data=df, x="region", y="customer_rating", hue="region", palette="Set3", legend=False, ax=axes[1])
axes[1].set_title("Customer Rating by Region")
save("06_box_profit_rating")

# 5.7 HEATMAP - correlations
corr = df[num_cols + ["profit_margin_pct"]].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=.5, cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Correlation Heatmap"); ax.grid(False)
save("07_heatmap_correlation")

# =============================================================== 6. INSIGHTS
print("\n" + "=" * 60 + "\n4. KEY INSIGHTS (auto-computed)\n" + "=" * 60)
ins = []
top = by_cat["revenue"].idxmax()
ins.append(f"{top} drives the most revenue (${by_cat['revenue'].max()/1e3:,.0f}K, "
           f"{by_cat['revenue'].max()/by_cat['revenue'].sum():.0%} of total).")
best_m = by_cat["margin_%"].idxmax()
ins.append(f"{best_m} has the best profit margin ({by_cat['margin_%'].max():.1f}%); "
           f"{by_cat['margin_%'].idxmin()} the worst ({by_cat['margin_%'].min():.1f}%).")
ins.append(f"Revenue peaks in {df.groupby('month_name')['revenue'].sum().idxmax()}; "
           f"Nov-Dec account for {df[df['order_date'].dt.month.isin([11, 12])]['revenue'].sum()/df['revenue'].sum():.0%} of revenue.")
ins.append(f"High discounts (20-30%) average {by_disc.loc['High (20-30%)','avg_profit']:+.2f} USD profit/order vs "
           f"{by_disc.loc['No discount','avg_profit']:+.2f} USD with no discount.")
r_ship = df["shipping_days"].corr(df["customer_rating"]); r_disc = df["discount"].corr(df["customer_rating"])
ins.append(f"Shipping time vs rating r = {r_ship:.2f}; discount vs rating r = {r_disc:.2f}.")
ins.append(f"South region is slowest to ship ({by_region.loc['South','avg_ship']:.1f} days vs "
           f"{by_region.drop('South')['avg_ship'].mean():.1f} elsewhere).")
ins.append(f"{log['quantity_outliers_flagged']} bulk-order outliers flagged; revenue is right-skewed "
           f"(skew {df['revenue'].skew():.1f}), so median (${df['revenue'].median():,.0f}) beats mean "
           f"(${df['revenue'].mean():,.0f}) as a typical-order measure.")
for i in ins:
    print(" -", i)
Path("insights.txt").write_text("\n".join(f"- {i}" for i in ins))
print(f"\nSaved 7 figures to ./{FIG}/, clean data to data/retail_orders_clean.csv")
