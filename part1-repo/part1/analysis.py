"""
Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis
Dataset: Synthetic online retail order-line data (raw_online_retail.csv)

Run top-to-bottom. Produces:
  - cleaned_data.csv
  - plots/ (7 PNG files: line, bar, hist, scatter, box, heatmap)
  - console output used to populate README.md
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ----------------------------------------------------------------------
# Task 1: Load data
# ----------------------------------------------------------------------
section("TASK 1: Load dataset")
df = pd.read_csv("raw_online_retail.csv")
print("First 5 rows:")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nShape:", df.shape)

# ----------------------------------------------------------------------
# Task 2: Null value analysis
# ----------------------------------------------------------------------
section("TASK 2: Null value analysis")
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct.round(2)})
null_table = null_table.sort_values("null_pct", ascending=False)
print(null_table)

high_null_cols = null_table[null_table["null_pct"] > 20].index.tolist()
print("\nColumns exceeding 20% null rate:", high_null_cols)

low_null_numeric = [
    c for c in df.select_dtypes(include=np.number).columns
    if c not in high_null_cols and df[c].isnull().sum() > 0
]
print("Numeric columns below 20% nulls to be median-filled:", low_null_numeric)

for col in low_null_numeric:
    df[col] = df[col].fillna(df[col].median())

print("\nNulls remaining after median-fill of low-null numeric columns:")
print(df.isnull().sum())

# ----------------------------------------------------------------------
# Task 3: Duplicate detection
# ----------------------------------------------------------------------
section("TASK 3: Duplicate detection and removal")
n_dupes = df.duplicated().sum()
print("Duplicate rows found:", n_dupes)

null_pct_before_dedup = (df.isnull().sum() / df.shape[0] * 100).round(3)
df = df.drop_duplicates()
null_pct_after_dedup = (df.isnull().sum() / df.shape[0] * 100).round(3)

print("Rows removed:", n_dupes)
print("New shape:", df.shape)
print("\nNull % before dedup vs after dedup:")
compare = pd.DataFrame({"before": null_pct_before_dedup, "after": null_pct_after_dedup})
print(compare)

# ----------------------------------------------------------------------
# Task 4: Data type correction
# ----------------------------------------------------------------------
section("TASK 4: Data type correction")
mem_before = df.memory_usage(deep=True).sum()
print("Memory usage BEFORE conversion (bytes):", mem_before)

# unit_price is object (currency strings + 'N/A') -> should be numeric
df["unit_price"] = df["unit_price"].replace("N/A", np.nan)
df["unit_price"] = df["unit_price"].str.replace("$", "", regex=False)
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
# Fill the newly created unit_price nulls with median (below 20% threshold check)
up_null_pct = df["unit_price"].isnull().mean() * 100
print(f"unit_price null % after coercion: {up_null_pct:.2f}%")
if up_null_pct <= 20:
    df["unit_price"] = df["unit_price"].fillna(df["unit_price"].median())

# product_category, customer_segment, region: repetitive strings -> category dtype
for col in ["product_category", "customer_segment", "region"]:
    df[col] = df[col].astype("category")

mem_after = df.memory_usage(deep=True).sum()
print("Memory usage AFTER conversion (bytes):", mem_after)
print(f"Memory reduction: {mem_before - mem_after} bytes "
      f"({(mem_before - mem_after) / mem_before * 100:.2f}%)")
print("\nUpdated dtypes:")
print(df.dtypes)

# ----------------------------------------------------------------------
# Task 5: Descriptive statistics and skewness
# ----------------------------------------------------------------------
section("TASK 5: Descriptive statistics and skewness")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "order_id"]
print(df[numeric_cols].describe())

skew_vals = df[numeric_cols].skew().sort_values(key=lambda s: s.abs(), ascending=False)
print("\nSkewness (sorted by |skew|):")
print(skew_vals)

most_skewed_col = skew_vals.index[0]
second_skewed_col = skew_vals.index[1]
print(f"\nMost skewed column: {most_skewed_col} (skew={skew_vals.iloc[0]:.3f})")
print(f"Second most skewed column: {second_skewed_col} (skew={skew_vals.iloc[1]:.3f})")

# ----------------------------------------------------------------------
# Task 6: Outlier detection with IQR
# ----------------------------------------------------------------------
section("TASK 6: Outlier detection with IQR")
iqr_cols = ["revenue", "quantity", "delivery_days"]
iqr_report = {}
for col in iqr_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    iqr_report[col] = dict(Q1=Q1, Q3=Q3, IQR=IQR, lower=lower, upper=upper, n_outliers=n_out)
    print(f"{col}: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}, "
          f"bounds=[{lower:.2f}, {upper:.2f}], outliers={n_out}")

# ----------------------------------------------------------------------
# Task 7: Visualizations
# ----------------------------------------------------------------------
section("TASK 7: Visualizations")

# 7a. Line plot
df_sorted = df.sort_values("order_date")
plt.figure(figsize=(10, 5))
plt.plot(df_sorted["order_date"], df_sorted["revenue"], linewidth=0.6, alpha=0.8)
plt.title("Revenue Over Time (per order line)")
plt.xlabel("Order Date")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig("plots/01_line_revenue_over_time.png", dpi=110)
plt.close()

# 7b. Bar chart: mean revenue by product_category
plt.figure(figsize=(8, 5))
cat_means = df.groupby("product_category", observed=True)["revenue"].mean().sort_values(ascending=False)
plt.bar(cat_means.index, cat_means.values, color="#4C72B0")
plt.title("Mean Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Mean Revenue ($)")
plt.tight_layout()
plt.savefig("plots/02_bar_mean_revenue_by_category.png", dpi=110)
plt.close()

# 7c. Histogram of most skewed column
plt.figure(figsize=(8, 5))
sns.histplot(df[most_skewed_col], bins=20, kde=True, color="#DD8452")
plt.title(f"Distribution of {most_skewed_col} (skew={skew_vals.iloc[0]:.2f})")
plt.xlabel(most_skewed_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/03_histogram_most_skewed.png", dpi=110)
plt.close()

# 7d. Scatter plot: quantity vs revenue (expected correlated)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="quantity", y="revenue", alpha=0.5, hue="product_category")
plt.title("Quantity vs Revenue")
plt.xlabel("Quantity")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig("plots/04_scatter_quantity_vs_revenue.png", dpi=110)
plt.close()

# 7e. Box plot: revenue by region
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="region", y="revenue")
plt.title("Revenue Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.savefig("plots/05_boxplot_revenue_by_region.png", dpi=110)
plt.close()

# 7f. Correlation heat map
plt.figure(figsize=(9, 7))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heat Map (Pearson)")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png", dpi=110)
plt.close()

# find highest abs correlation pair (excluding diagonal, upper triangle only to avoid mirror dupes)
upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_unstacked = corr_matrix.where(upper_mask).unstack().dropna()
top_pair = corr_unstacked.abs().sort_values(ascending=False).index[0]
top_pair_val = corr_matrix.loc[top_pair[0], top_pair[1]]
print(f"Highest |correlation| pair: {top_pair} = {top_pair_val:.3f}")

print("Saved 6 plot files to plots/")

# ----------------------------------------------------------------------
# Task 8a: Imputation strategy comparison (mean vs median) for two most-skewed cols
# ----------------------------------------------------------------------
section("TASK 8a: Imputation strategy comparison (mean vs median)")
# Reload raw values pre-imputation for these two specific columns to compare honestly
df_raw = pd.read_csv("raw_online_retail.csv")
df_raw["unit_price"] = pd.to_numeric(df_raw["unit_price"].replace("N/A", np.nan).str.replace("$", "", regex=False), errors="coerce")

target_cols = [most_skewed_col, second_skewed_col]
compare_stats = {}
for col in target_cols:
    src = df_raw[col] if col in df_raw.columns else df[col]
    mean_v = src.mean()
    median_v = src.median()
    compare_stats[col] = {"mean": mean_v, "median": median_v}
    print(f"{col}: mean={mean_v:.3f}, median={median_v:.3f}")

for col in target_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\nNulls remaining in target columns after chosen imputation:")
print(df[target_cols].isnull().sum())

# ----------------------------------------------------------------------
# Task 8b: Spearman vs Pearson
# ----------------------------------------------------------------------
section("TASK 8b: Spearman rank correlation vs Pearson")
pearson_matrix = df[numeric_cols].corr(method="pearson")
spearman_matrix = df[numeric_cols].corr(method="spearman")
print("Pearson matrix:")
print(pearson_matrix.round(3))
print("\nSpearman matrix:")
print(spearman_matrix.round(3))

diff_matrix = (spearman_matrix - pearson_matrix).abs()
upper_mask2 = np.triu(np.ones(diff_matrix.shape), k=1).astype(bool)
diff_unstacked = diff_matrix.where(upper_mask2).unstack().dropna()
top3_pairs = diff_unstacked.sort_values(ascending=False).head(3)

print("\nTop 3 pairs by |Spearman - Pearson|:")
diff_table_rows = []
for (a, b), diff in top3_pairs.items():
    p = pearson_matrix.loc[a, b]
    s = spearman_matrix.loc[a, b]
    diff_table_rows.append({"pair": f"{a} vs {b}", "pearson": round(p, 3),
                             "spearman": round(s, 3), "abs_diff": round(diff, 3)})
diff_table = pd.DataFrame(diff_table_rows)
print(diff_table)

# ----------------------------------------------------------------------
# Task 8c: Grouped aggregation
# ----------------------------------------------------------------------
section("TASK 8c: Grouped aggregation")
group_col = "region"
agg_col = "revenue"
grouped = df.groupby(group_col, observed=True)[agg_col].agg(["mean", "std", "count"])
print(grouped)

highest_mean_group = grouped["mean"].idxmax()
highest_std_group = grouped["std"].idxmax()
mean_ratio = grouped["mean"].max() / grouped["mean"].min()
print(f"\nHighest mean group: {highest_mean_group} ({grouped['mean'].max():.2f})")
print(f"Highest std group: {highest_std_group} ({grouped['std'].max():.2f})")
print(f"Ratio highest-mean/lowest-mean: {mean_ratio:.3f}")

# ----------------------------------------------------------------------
# Task 9: Save cleaned dataset
# ----------------------------------------------------------------------
section("TASK 9: Save cleaned dataset")
df.to_csv("cleaned_data.csv", index=False)
print("Saved cleaned_data.csv, shape:", df.shape)
print("\nFinal null check:")
print(df.isnull().sum())

section("DONE")
