from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "retail_sales_raw.csv")
VIZ = BASE / "visualizations"
REP = BASE / "reports"
VIZ.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

num = ["Quantity", "Sales", "Discount", "Profit", "Shipping_Days"]

df[num].describe().T.to_csv(REP/"descriptive_statistics.csv")
df.isna().sum().to_frame("Missing_Count").to_csv(REP/"missing_values.csv")

outliers = []
for col in num:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    count = ((df[col] < low) | (df[col] > high)).sum()
    outliers.append([col, q1, q3, low, high, count])
pd.DataFrame(outliers, columns=["Feature","Q1","Q3","Lower_Bound","Upper_Bound","Outlier_Count"]).to_csv(REP/"outlier_analysis.csv", index=False)

for name, col in [
    ("category_summary.csv","Category"),
    ("region_summary.csv","Region"),
    ("segment_summary.csv","Segment"),
    ("city_summary.csv","City")
]:
    df.groupby(col).agg(Orders=("Order_ID","count"), Sales=("Sales","sum"), Profit=("Profit","sum")).sort_values("Sales", ascending=False).to_csv(REP/name)

corr = df[num].corr()
corr.to_csv(REP/"correlation_matrix.csv")

plots = [
    ("01_sales_by_category.png", df.groupby("Category")["Sales"].sum().sort_values(), "Total Sales by Category", "barh"),
    ("02_profit_by_region.png", df.groupby("Region")["Profit"].sum().sort_values(), "Total Profit by Region", "barh"),
    ("03_sales_by_segment.png", df.groupby("Segment")["Sales"].sum().sort_values(), "Total Sales by Customer Segment", "bar"),
]
for filename, series, title, kind in plots:
    plt.figure(figsize=(9,6))
    series.plot(kind=kind, title=title)
    plt.tight_layout()
    plt.savefig(VIZ/filename, dpi=150)
    plt.close()

monthly = df.groupby("Month")["Sales"].sum()
plt.figure(figsize=(12,6)); monthly.plot(marker="o", title="Monthly Sales Trend")
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(VIZ/"04_monthly_sales_trend.png", dpi=150); plt.close()

monthly_profit = df.groupby("Month")["Profit"].sum()
plt.figure(figsize=(12,6)); monthly_profit.plot(marker="o", title="Monthly Profit Trend")
plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(VIZ/"05_monthly_profit_trend.png", dpi=150); plt.close()

plt.figure(figsize=(9,6)); sns.scatterplot(data=df, x="Sales", y="Profit", hue="Category", alpha=.65)
plt.title("Sales vs Profit"); plt.tight_layout(); plt.savefig(VIZ/"06_sales_vs_profit.png", dpi=150); plt.close()

plt.figure(figsize=(9,6)); sns.scatterplot(data=df, x="Discount", y="Profit", hue="Category", alpha=.65)
plt.title("Discount vs Profit"); plt.tight_layout(); plt.savefig(VIZ/"07_discount_vs_profit.png", dpi=150); plt.close()

plt.figure(figsize=(9,7)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True)
plt.title("Correlation Matrix"); plt.tight_layout(); plt.savefig(VIZ/"08_correlation_heatmap.png", dpi=150); plt.close()

for i, col in enumerate(num, 9):
    plt.figure(figsize=(9,5)); plt.hist(df[col].dropna(), bins=20, edgecolor="black")
    plt.title("Distribution of " + col.replace("_"," "))
    plt.tight_layout(); plt.savefig(VIZ/f"{i:02d}_distribution_{col.lower()}.png", dpi=150); plt.close()

top_city = df.groupby("City")["Sales"].sum().sort_values().tail(10)
plt.figure(figsize=(10,6)); top_city.plot(kind="barh", title="Top 10 Cities by Sales")
plt.tight_layout(); plt.savefig(VIZ/"14_top_10_cities_by_sales.png", dpi=150); plt.close()

report = (
    "EXPLORATORY DATA ANALYSIS REPORT\n\n"
    f"Rows: {len(df):,}\n"
    f"Columns: {df.shape[1]}\n"
    f"Duplicate rows: {df.duplicated().sum()}\n"
    f"Missing cells: {df.isna().sum().sum()}\n\n"
    "KEY FINDINGS\n"
    f"Highest-sales category: {df.groupby('Category')['Sales'].sum().idxmax()}\n"
    f"Highest-sales region: {df.groupby('Region')['Sales'].sum().idxmax()}\n"
    f"Highest-profit region: {df.groupby('Region')['Profit'].sum().idxmax()}\n"
    f"Highest-sales segment: {df.groupby('Segment')['Sales'].sum().idxmax()}\n"
    f"Highest-sales city: {df.groupby('City')['Sales'].sum().idxmax()}\n"
    f"Sales-Profit correlation: {corr.loc['Sales','Profit']:.2f}\n"
    f"Discount-Profit correlation: {corr.loc['Discount','Profit']:.2f}\n"
)
(REP/"EDA_Report.txt").write_text(report, encoding="utf-8")
