"""
Generates a synthetic, intentionally messy "online retail orders" dataset
for the data cleaning / EDA / visualization assignment.

Domain: Online retail order-line data. Each row is one product line
within a customer order.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 2000

# ---- Base dimensions -------------------------------------------------
order_id = np.arange(100000, 100000 + N)

order_date = pd.date_range("2024-01-01", periods=N, freq="4h")

customer_segment = rng.choice(
    ["Consumer", "Corporate", "Home Office"], size=N, p=[0.55, 0.30, 0.15]
)

region = rng.choice(
    ["North", "South", "East", "West", "Central"], size=N,
    p=[0.22, 0.22, 0.22, 0.22, 0.12]
)

product_category = rng.choice(
    ["Electronics", "Furniture", "Office Supplies", "Clothing"],
    size=N, p=[0.30, 0.20, 0.35, 0.15]
)

# Region has a genuine effect baked in on shipping cost / delivery days
region_delivery_bump = {"North": 0, "South": 1, "East": 2, "West": 3, "Central": 1}

# Category base price ranges (drives skew + correlation with revenue)
category_base_price = {
    "Electronics": 220, "Furniture": 340, "Office Supplies": 18, "Clothing": 45
}

quantity = rng.integers(1, 6, size=N).astype(float)

# unit price: lognormal -> positively skewed, tied to category
base_price = np.array([category_base_price[c] for c in product_category])
unit_price = base_price * rng.lognormal(mean=0.0, sigma=0.55, size=N)
unit_price = np.round(unit_price, 2)

discount_pct = np.round(rng.beta(2, 8, size=N), 3)  # mostly small discounts

# revenue: genuinely correlated with quantity and unit_price (with discount applied)
revenue = np.round(quantity * unit_price * (1 - discount_pct), 2)

# rating 1-5, slightly negatively skewed (most people rate high)
rating = np.clip(rng.normal(4.2, 0.9, size=N), 1, 5)
rating = np.round(rating, 1)

# delivery days: depends on region + some noise, positively skewed with a long tail
delivery_days = np.array([
    max(1, int(rng.normal(3 + region_delivery_bump[r], 1.2)))
    for r in region
]).astype(float)
# inject some real long-tail delays (outliers)
delay_idx = rng.choice(N, size=25, replace=False)
delivery_days[delay_idx] += rng.integers(10, 25, size=25)

# shipping cost, loosely tied to weight/category, will get heavy nulls (>20%)
shipping_cost = np.round(np.clip(rng.normal(8, 4, size=N), 1, None) +
                          (product_category == "Furniture") * 15, 2)

# customer age, minor missingness (<20%)
customer_age = rng.integers(18, 75, size=N).astype(float)

df = pd.DataFrame({
    "order_id": order_id,
    "order_date": order_date,
    "customer_segment": customer_segment,
    "region": region,
    "product_category": product_category,
    "customer_age": customer_age,
    "quantity": quantity,
    "unit_price": unit_price,
    "discount_pct": discount_pct,
    "revenue": revenue,
    "rating": rating,
    "delivery_days": delivery_days,
    "shipping_cost": shipping_cost,
})

# ---- Inject messiness --------------------------------------------------

# 1. Heavy missingness in shipping_cost (>20%)
mask = rng.random(N) < 0.28
df.loc[mask, "shipping_cost"] = np.nan

# 2. Light missingness (<20%) in several numeric columns
for col, p in [("customer_age", 0.06), ("rating", 0.09),
               ("delivery_days", 0.05), ("revenue", 0.04)]:
    mask = rng.random(N) < p
    df.loc[mask, col] = np.nan

# 3. Outliers deliberately left in revenue and quantity
extreme_idx = rng.choice(N, size=15, replace=False)
df.loc[extreme_idx, "revenue"] = df.loc[extreme_idx, "revenue"] * rng.uniform(6, 12, size=15)
big_qty_idx = rng.choice(N, size=10, replace=False)
df.loc[big_qty_idx, "quantity"] = rng.integers(25, 50, size=10)

# 4. Duplicate rows (exact copies) injected
dup_rows = df.sample(40, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

# 5. unit_price wrongly typed as object (currency strings + stray blanks)
up_str = df["unit_price"].map(lambda x: f"${x:.2f}")
noise_idx = rng.choice(len(df), size=20, replace=False)
up_str.iloc[noise_idx] = "N/A"
df["unit_price"] = up_str  # now dtype object, needs cleaning -> to_numeric

# 6. product_category left as plain object but is highly repetitive -> category dtype target

# Shuffle rows so duplicates aren't all at the tail (more realistic "raw file")
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

df.to_csv("raw_online_retail.csv", index=False)
print("Saved raw_online_retail.csv with shape:", df.shape)
print(df.dtypes)
