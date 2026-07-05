# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset Description

`raw_online_retail.csv` is a synthetic **online retail order-line dataset**: each row is one
product line within a customer order. It was generated for this assignment (2,040 raw rows,
13 columns) with realistic messiness deliberately injected so that every required cleaning
and EDA technique has something genuine to catch — missing values at varying rates,
duplicate rows, a numeric column mis-typed as text, skewed distributions, and real outliers.

| Column | Description |
|---|---|
| `order_id` | Unique order-line identifier |
| `order_date` | Timestamp of the order |
| `customer_segment` | Consumer / Corporate / Home Office |
| `region` | North / South / East / West / Central |
| `product_category` | Electronics / Furniture / Office Supplies / Clothing |
| `customer_age` | Customer age in years |
| `quantity` | Units purchased in this order line |
| `unit_price` | Price per unit ($) |
| `discount_pct` | Discount applied (fraction) |
| `revenue` | `quantity × unit_price × (1 − discount_pct)`, with real-world noise/outliers added |
| `rating` | Customer satisfaction rating (1–5) |
| `delivery_days` | Days from order to delivery |
| `shipping_cost` | Shipping cost ($) |

Why this dataset: it naturally contains a categorical variable with several groups
(`region`, `product_category`) for the groupby/box-plot tasks, at least two skewed numeric
columns (`quantity`, `revenue`), a genuinely correlated numeric pair
(`unit_price`/`quantity` → `revenue`), and columns with different missingness levels — so
each part of the spec maps onto a real feature of the data rather than an artificial example.

## Files

- `generate_data.py` — generates the raw synthetic dataset (`raw_online_retail.csv`)
- `data_cleaning_eda.ipynb` — notebook with all cleaning/EDA/visualization code (outputs cleared; re-run top-to-bottom to reproduce, also mirrored in `analysis.py` as a plain script)
- `analysis.py` — equivalent plain-Python script version of the notebook
- `raw_online_retail.csv` — the raw dataset (input)
- `cleaned_data.csv` — the cleaned output dataset (produced by the code, committed for convenience)
- `plots/` — all 6 saved chart images, produced via `plt.savefig()` in the code
- `requirements.txt` — Python dependencies
- `analysis_output.txt` — full console output from a reference run of `analysis.py`

## How to Run

**Dependencies** (also in `requirements.txt`): `pandas`, `numpy`, `matplotlib`, `seaborn`.
No API keys, credentials, or environment variables are required anywhere in this part.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the raw dataset from scratch — raw_online_retail.csv is
#    already included in this repo, so this step is only needed if you want to
#    regenerate it or vary the random seed
python3 generate_data.py

# 3. Run the full cleaning/EDA/visualization pipeline
python3 analysis.py
# writes cleaned_data.csv and plots/*.png, prints all findings to the console

# OR, equivalently, open and run data_cleaning_eda.ipynb top-to-bottom
# (Jupyter / VS Code / Google Colab all work — no environment-specific code is used)
```

## Step-by-Step Findings

### Task 1 — Load
Loaded with `pd.read_csv()`. Raw shape: **(2040, 13)**. Several columns loaded with
incorrect dtypes at this stage (`unit_price` as text, categorical columns as generic
strings) — addressed in Task 4.

### Task 2 — Null value analysis

| Column | Null % |
|---|---|
| shipping_cost | 29.12% |
| rating | 9.07% |
| customer_age | 5.29% |
| delivery_days | 4.71% |
| revenue | 4.36% |
| unit_price | 0.98% |
| (all others) | 0.00% |

**Column exceeding 20%:** `shipping_cost`. It is **not** median-filled — imputing nearly a
third of a column with one statistic would meaningfully distort its distribution, so it's
documented here and left as a deliberate decision for Part 2 (e.g., a missingness indicator
flag or a model-based imputation).

**Columns below 20% median-filled:** `customer_age`, `rating`, `delivery_days`, `revenue`.

**Why median instead of mean?** Two of these columns (`revenue`, and by extension the
correlated `quantity`) are strongly right-skewed (see Task 5). The mean of a skewed column is
pulled toward its long tail by a small number of extreme values, so it no longer represents a
"typical" row. The median is robust to that pull, making it the safer fill value across all
four columns (kept consistent rather than mixing strategies per-column).

### Task 3 — Duplicates
**40 duplicate rows** found and removed (`df.duplicated().sum()` → drop_duplicates).
New shape: **(2000, 13)**. Comparing null percentages before/after removal shows almost no
change (e.g. `shipping_cost`: 29.118% → 29.1%) — the duplicated rows carried a representative
mix of missingness, so dropping them didn't skew the overall null picture.

### Task 4 — Data type correction
`unit_price` was loaded as an `object`/string column because the raw export mixed a `$`
prefix with literal `"N/A"` placeholders. It was cleaned (`$` stripped, `"N/A"` → `NaN`) and
converted with `pd.to_numeric(errors="coerce")`. The three repetitive text columns
(`product_category`, `customer_segment`, `region`) were converted to pandas `category` dtype.

**Memory usage:** 734,807 bytes → 302,684 bytes — a **58.8% reduction**, driven mostly by the
category conversions (each unique string stored once instead of per-row) plus `unit_price`
becoming a compact numeric column instead of Python objects.

### Task 5 — Descriptive statistics and skewness
`df.describe()` was run on all 8 numeric columns. Skewness (sorted by |skew|):

| Column | Skew |
|---|---|
| quantity | **10.02** |
| revenue | **8.04** |
| delivery_days | 4.28 |
| unit_price | 2.01 |
| shipping_cost | 0.92 |
| discount_pct | 0.88 |
| rating | −0.76 |
| customer_age | 0.04 |

**Most skewed column: `quantity`** (skew ≈ 10.0), closely followed by `revenue` (≈ 8.0).
Both are **strongly positively skewed**: the bulk of orders sit in a small, tight range
(1–5 units, modest revenue), while a small number of bulk orders (25–50 units) and
high-ticket transactions stretch a long tail out to the right. (`rating`, in contrast, is
mildly *negatively* skewed — most ratings cluster high, with a shorter tail toward low
scores.)

**Consequence for mean-imputation:** because the mean is dragged toward the long right tail,
filling missing values in `quantity` or `revenue` with the column *mean* would insert values
noticeably larger than what a typical order actually looks like. This is exactly why the
median was used for these columns in Tasks 2 and 8a.

### Task 6 — Outlier detection (IQR)

| Column | Q1 | Q3 | IQR | Bounds | Outliers |
|---|---|---|---|---|---|
| revenue | 52.66 | 523.61 | 470.94 | [−653.75, 1230.03] | **149** |
| quantity | 2.00 | 4.00 | 2.00 | [−1.00, 7.00] | **10** |
| delivery_days | 3.00 | 5.00 | 2.00 | [0.00, 8.00] | **27** |

None of these rows were dropped. All three look like genuine business events (large orders,
bulk purchases, real shipping delays) rather than data-entry errors, and removing them would
throw away exactly the signal a downstream model would want to learn. **Plan for Part 2:**
retain all rows, but **cap (winsorize) `revenue` and `quantity` at their upper IQR bound**
before feeding a linear/regression-style model (since a few extreme rows can dominate a
squared-error loss), while leaving `delivery_days` untouched if a tree-based model is used,
since those are not sensitive to this kind of outlier.

### Task 7 — Visualizations (all saved in `plots/`)

1. **Line plot** (`01_line_revenue_over_time.png`) — revenue per order line across the year.
   No strong trend; visible spikes line up with the large-ticket orders flagged as outliers.
2. **Bar chart** (`02_bar_mean_revenue_by_category.png`) — mean revenue by product category.
   Furniture and Electronics have the highest average revenue per line; Office Supplies the
   lowest, consistent with their underlying base price ranges.
3. **Histogram** (`03_histogram_most_skewed.png`) — distribution of `quantity`.
   **Shape:** a tall, tight cluster between 1–5 units with a thin, long right tail out to
   25–50 units — a textbook strongly right-skewed distribution.
4. **Scatter plot** (`04_scatter_quantity_vs_revenue.png`) — `quantity` vs `revenue`.
   **Interpretation:** a visible but only moderate positive relationship (Pearson r ≈ 0.12,
   Spearman ≈ 0.33 — see Task 8b); revenue is driven at least as much by `unit_price`, so the
   same quantity lands at very different revenue depending on product category.
5. **Box plot** (`05_boxplot_revenue_by_region.png`) — `revenue` split by `region`.
   **Interpretation:** median revenue is very similar across all five regions, but spread
   differs sharply — North has by far the widest box and most extreme high-end outliers,
   meaning revenue there is much less predictable than in, say, Central (confirmed by the
   grouped standard deviations in Task 8c).
6. **Correlation heat map** (`06_correlation_heatmap.png`) — Pearson correlations across all
   numeric columns. **Highest |correlation| pair: `unit_price` & `revenue` (r ≈ 0.65).**
   This is largely a *mechanical* relationship rather than a coincidental one — `revenue` is
   literally computed as `quantity × unit_price × (1 − discount)`, so `unit_price` is a direct
   input, not just an associated variable. If this correlation were observed without knowing
   the formula, one plausible **alternative (confounding) explanation** would be a hidden
   "product tier" factor: premium products tend to carry both a higher unit price *and*
   independently higher demand or larger typical order sizes, which could inflate the observed
   correlation beyond the pure price → revenue arithmetic.

### Task 8a — Imputation strategy comparison (mean vs. median)

| Column | Mean | Median |
|---|---|---|
| quantity | 3.127 | 3.000 |
| revenue | 432.114 | 166.310 |

For both columns the mean sits well above the median — the fingerprint of positive skew
pulling the mean upward via the bulk-order / high-ticket tail. **Chosen strategy: median**,
for both columns, since it represents the typical order far better than a mean inflated by a
small number of large orders. After imputation, `isnull().sum()` confirms **zero** remaining
nulls in `quantity` and `revenue`.

### Task 8b — Spearman vs. Pearson

Top 3 pairs by `|Spearman − Pearson|`:

| Pair | Pearson | Spearman | \|diff\| |
|---|---|---|---|
| revenue vs unit_price | 0.650 | 0.883 | 0.232 |
| revenue vs quantity | 0.120 | 0.332 | 0.212 |
| shipping_cost vs unit_price | 0.495 | 0.388 | 0.107 |

**Interpretation:**
1. **`revenue` vs `unit_price`** — |Spearman| > |Pearson|: **monotonic but non-linear**.
   Expected, since `revenue` is a *product* involving `unit_price`, so higher unit price
   reliably raises revenue in rank order without a fixed linear step size.
2. **`revenue` vs `quantity`** — |Spearman| > |Pearson|: also **monotonic but non-linear**.
   Pearson is muted by a few extreme high-revenue/low-quantity rows and the bulk-order
   outliers pulling against a straight-line fit, while the rank relationship (more units →
   more revenue) holds up more consistently.
3. **`shipping_cost` vs `unit_price`** — here |Pearson| ≥ |Spearman|: this relationship is
   **closer to linear** than the other two; the modest drop in Spearman suggests only a small
   departure from strict monotonicity.

**Which measure guides Part 2 feature selection: Spearman**, at least for the
revenue-related features. Since the two most important relationships to `revenue` are
monotonic-but-non-linear, Pearson would understate their usefulness for a non-linear or
tree-based model, and Spearman is also more robust to the outliers documented in Task 6.

### Task 8c — Grouped aggregation (`region` × `revenue`)

| Region | Mean | Std | Count |
|---|---|---|---|
| Central | 417.19 | 628.46 | 235 |
| East | 425.23 | 602.91 | 442 |
| North | 422.79 | **902.53** | 453 |
| South | **430.72** | 733.90 | 446 |
| West | 404.66 | 737.02 | 424 |

- **Highest mean:** South ($430.72). **Highest std dev:** North ($902.53).
- **Is high within-group std a concern?** Yes — North's standard deviation (≈$903) is more
  than double its own mean (≈$423), meaning revenue outcomes within North swing enormously.
  Knowing an order shipped from North tells a model very little about what its revenue will
  actually be.
- **Mean ratio (highest ÷ lowest) = 1.064** — group means are within ~6% of each other. A
  ratio this close to 1 indicates `region` alone carries **very little predictive signal**
  for revenue; the real explanatory power lives in `unit_price`, `quantity`, and
  `product_category`, not shipping region.

### Task 9 — Cleaned dataset
Saved as `cleaned_data.csv`, final shape **(2000, 13)**. `shipping_cost` still contains
nulls by design (582 remaining) since it exceeded the 20% threshold and its treatment is
deferred to Part 2, as documented in Task 2.

## Plot-by-Plot Summary

All plots are produced by `plt.savefig()` calls in `analysis.py` / `data_cleaning_eda.ipynb`
and committed under `plots/`:

| File | Type | What it shows |
|---|---|---|
| `01_line_revenue_over_time.png` | Line plot | Revenue per order line across the year; no strong trend, spikes match large-ticket outlier orders |
| `02_bar_mean_revenue_by_category.png` | Bar chart | Mean revenue per product category; Furniture/Electronics highest, Office Supplies lowest |
| `03_histogram_most_skewed.png` | Histogram | Distribution of `quantity` (most skewed column, skew ≈10) — tight cluster near 1–5 units, long right tail to 25–50 |
| `04_scatter_quantity_vs_revenue.png` | Scatter plot | `quantity` vs `revenue`, colored by product category — moderate positive relationship |
| `05_boxplot_revenue_by_region.png` | Box plot | `revenue` by `region` — similar medians, but North has much wider spread and more extreme outliers |
| `06_correlation_heatmap.png` | Heat map | Pearson correlation across all numeric columns — `unit_price`/`revenue` is the strongest pair (r≈0.65) |

## Security / Secrets

This part uses only local CSV files and open-source Python libraries. **No API keys,
credentials, or environment variables are used or required anywhere in this repository.**

