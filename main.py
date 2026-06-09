import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', 'Arial Unicode MS']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.edgecolor'] = '#333333'

np.random.seed(2026)
base = "./caravan-qual-csv/wqms-csv"
plot_dir = "./plots"
os.makedirs(plot_dir, exist_ok=True)

# count observations per constituent
counts = {}
for fname in os.listdir(base):
    if not fname.endswith(".csv"):
        continue
    name = fname.replace(".csv", "")
    counts[name] = sum(1 for _ in open(os.path.join(base, fname))) - 1

df_counts = pd.DataFrame(list(counts.items()), columns=["constituent", "n_obs"])
df_counts = df_counts.sort_values("n_obs", ascending=False).reset_index(drop=True)
top25 = df_counts.head(25).copy()

# fig: observation count top 25
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top25)), top25["n_obs"].values, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(top25)))
ax.set_yticklabels(top25["constituent"].values)
ax.invert_yaxis()
ax.set_xlabel("Number of Observations")
ax.set_title("Top 25 Water Quality Constituents by Observation Count")
ax.ticklabel_format(axis="x", style="scientific", scilimits=(6, 6))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "obs_count_top25.png"), dpi=300, bbox_inches="tight")
plt.close()

print("Top 25 constituents:")
for _, row in top25.iterrows():
    print(f"  {row['constituent']:<20s} {row['n_obs']:>10,d}")
print(f"Total: {df_counts['n_obs'].sum():,d}")

# fig: streamflow coverage top 25
sf_cov = {}
for name in top25["constituent"].values:
    df_s = pd.read_csv(os.path.join(base, f"{name}.csv"), usecols=["streamflow"], nrows=50000)
    sf_cov[name] = df_s["streamflow"].notna().mean() * 100

sf_df = pd.DataFrame({"constituent": list(sf_cov.keys()), "pct": list(sf_cov.values())})
sf_order = {c: i for i, c in enumerate(top25["constituent"].values)}
sf_df["order"] = sf_df["constituent"].map(sf_order)
sf_df = sf_df.sort_values("order")

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#C44E52" if v < 10 else "#4C72B0" for v in sf_df["pct"].values]
ax.barh(range(len(sf_df)), sf_df["pct"].values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(sf_df)))
ax.set_yticklabels(sf_df["constituent"].values)
ax.invert_yaxis()
ax.set_xlabel("Observations with Paired Streamflow (%)")
ax.set_title("Streamflow Data Availability for Top 25 Constituents")
ax.axvline(x=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "streamflow_coverage.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nStreamflow coverage:")
for _, row in sf_df.iterrows():
    print(f"  {row['constituent']:<20s} {row['pct']:>6.1f}%")

# fig: temporal distribution
temporal_src = ["DO", "NO3N", "TP", "NH4N", "BOD5"]
all_years = []
for name in temporal_src:
    df_t = pd.read_csv(os.path.join(base, f"{name}.csv"), usecols=["dates"], nrows=300000)
    years = pd.to_datetime(df_t["dates"], errors="coerce").dt.year.dropna().astype(int)
    all_years.extend(years.tolist())

all_years = np.array(all_years)
all_years = all_years[(all_years >= 1980) & (all_years <= 2024)]

fig, ax = plt.subplots(figsize=(12, 5))
bins = np.arange(1980, 2026)
ax.hist(all_years, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Observations")
ax.set_title(f"Temporal Distribution of Water Quality Observations ({', '.join(temporal_src)})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "temporal_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

hist_vals, hist_edges = np.histogram(all_years, bins=bins)
print(f"\nTemporal distribution ({len(all_years):,d} sampled obs):")
for i in range(0, len(hist_vals), 5):
    chunk = hist_vals[i:i+5]
    yr_s, yr_e = int(hist_edges[i]), int(hist_edges[min(i+5, len(hist_edges)-1)]) - 1
    print(f"  {yr_s}-{yr_e}: {chunk.sum():>10,d}")

# fig: concentration-discharge scatter
cq_names = ["DO", "NO3N", "TP", "TSS"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
cq_stats = {}

for idx, name in enumerate(cq_names):
    df_cq = pd.read_csv(os.path.join(base, f"{name}.csv"), usecols=["obs", "streamflow", "unit"], nrows=200000)
    df_cq = df_cq.dropna(subset=["obs", "streamflow"])
    df_cq = df_cq[(df_cq["obs"] > 0) & (df_cq["streamflow"] > 0)]
    unit = df_cq["unit"].iloc[0] if len(df_cq) > 0 else ""
    corr = np.log10(df_cq["obs"]).corr(np.log10(df_cq["streamflow"]))
    cq_stats[name] = {"n": len(df_cq), "unit": unit, "corr": corr, "med": df_cq["obs"].median()}
    plot_df = df_cq.sample(min(5000, len(df_cq)), random_state=2026)
    ax = axes[idx]
    ax.scatter(plot_df["streamflow"], plot_df["obs"], s=3, alpha=0.15, color="#4C72B0", rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Streamflow (m$^3$/s)")
    ax.set_ylabel(f"{name} ({unit})")
    ax.set_title(f"{name} (n={len(df_cq):,d}, r={corr:+.2f})")

fig.suptitle("Concentration-Discharge Relationships", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "concentration_discharge.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nC-Q statistics:")
for name, s in cq_stats.items():
    print(f"  {name:<8s} n={s['n']:>6,d}  r={s['corr']:+.3f}  median={s['med']:.3f} {s['unit']}")

# fig: seasonal DO variation with streamflow
df_do = pd.read_csv(os.path.join(base, "DO.csv"), usecols=["wqms_id", "dates", "obs", "streamflow"])
df_do["dates"] = pd.to_datetime(df_do["dates"], errors="coerce")
df_do = df_do.dropna(subset=["dates", "obs", "streamflow"])
df_do = df_do[df_do["obs"] > 0]
df_do["month"] = df_do["dates"].dt.month

monthly = df_do.groupby("month").agg(do_med=("obs", "median"), sf_med=("streamflow", "median")).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(monthly["month"], monthly["do_med"], "o-", color="#4C72B0", linewidth=2, markersize=6, label="DO median")
ax1.set_xlabel("Month")
ax1.set_ylabel("Dissolved Oxygen (mg/L)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45)

ax2 = ax1.twinx()
ax2.plot(monthly["month"], monthly["sf_med"], "s--", color="#C44E52", linewidth=2, markersize=6, label="Streamflow median")
ax2.set_ylabel("Streamflow (m$^3$/s)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.set_title("Seasonal Variation of Dissolved Oxygen and Streamflow")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "seasonal_do.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSeasonal DO (n={len(df_do):,d} paired obs):")
for _, row in monthly.iterrows():
    print(f"  Month {int(row['month']):>2d}: DO={row['do_med']:.2f} mg/L  Q={row['sf_med']:.1f} m3/s")

# fig: single station paired time series
station_counts = df_do.groupby("wqms_id").size().reset_index(name="n")
best_station = station_counts.sort_values("n", ascending=False).iloc[0]["wqms_id"]
df_st = df_do[df_do["wqms_id"] == best_station].sort_values("dates")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df_st["dates"], df_st["obs"], "o", color="#4C72B0", markersize=3, alpha=0.7, label="DO")
ax1.set_xlabel("Date")
ax1.set_ylabel("Dissolved Oxygen (mg/L)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")

ax2 = ax1.twinx()
ax2.plot(df_st["dates"], df_st["streamflow"], "-", color="#C44E52", linewidth=0.8, alpha=0.6, label="Streamflow")
ax2.set_ylabel("Streamflow (m$^3$/s)", color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.set_title(f"Paired DO and Streamflow Time Series (Station: {best_station})")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "paired_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close()

date_range = f"{df_st['dates'].min().strftime('%Y-%m-%d')} to {df_st['dates'].max().strftime('%Y-%m-%d')}"
print(f"\nPaired time series (station {best_station}):")
print(f"  Observations: {len(df_st):,d}")
print(f"  Period: {date_range}")
print(f"  DO range: {df_st['obs'].min():.1f} to {df_st['obs'].max():.1f} mg/L")
print(f"  Q range: {df_st['streamflow'].min():.1f} to {df_st['streamflow'].max():.1f} m3/s")

print(f"\nAll 6 figures saved to {plot_dir}/")