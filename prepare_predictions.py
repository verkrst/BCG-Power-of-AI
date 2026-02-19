"""
Pre-compute barley yield predictions for all SSP scenarios and save to CSV.
Run once:  python prepare_predictions.py
"""

import unicodedata, re
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────
def norm_dep(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("'", "")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def max_consecutive_ones(x: pd.Series) -> int:
    run = best = 0
    for v in x.values:
        if v == 1:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
barley_path  = BASE / "dataset" / "barley_yield_from_1982.csv"
climate_path = BASE / "dataset" / "climate_data_from_1982.parquet"
out_path     = BASE / "dataset" / "predictions.csv"

# ── load data ────────────────────────────────────────────────────────────
print("Loading barley CSV …")
barley = pd.read_csv(barley_path, sep=";", index_col=0)
barley["dep_norm"] = barley["department"].map(norm_dep)

# fill missing yield where possible
mask = barley["yield"].isna() & barley["production"].notna() & (barley["area"] > 0)
barley.loc[mask, "yield"] = barley.loc[mask, "production"] / barley.loc[mask, "area"]

print("Loading climate parquet …")
climate_df = pd.read_parquet(climate_path)

# ── feature engineering ──────────────────────────────────────────────────
print("Pivoting climate data …")
climate = climate_df.copy()
climate["time"]  = pd.to_datetime(climate["time"], errors="coerce")
climate["year"]  = climate["time"].dt.year.astype("Int64")
climate["value"] = pd.to_numeric(climate["value"], errors="coerce")

w = (
    climate
    .pivot_table(
        index=["scenario", "code_dep", "nom_dep", "time", "year"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )
    .reset_index()
)
w.columns.name = None

# unit conversions: Kelvin → Celsius, mm/s → mm/day
for col in ["near_surface_air_temperature",
            "daily_maximum_near_surface_air_temperature"]:
    if col in w.columns:
        w[col] = w[col] - 273.15
if "precipitation" in w.columns:
    w["precipitation_mm_day"] = w["precipitation"] * 86400.0

# growing season (Mar–Jul) + stress indicators
GROW_START, GROW_END = 3, 7
TBASE, HEAT_TH, DRY_TH = 5.0, 30.0, 1.0

w["month"] = w["time"].dt.month
gs = w[(w["month"] >= GROW_START) & (w["month"] <= GROW_END)].copy()

gs["hot_day"] = (gs["daily_maximum_near_surface_air_temperature"] > HEAT_TH).astype(int)
gs["dry_day"] = (gs["precipitation_mm_day"] < DRY_TH).astype(int)
gs["gdd_day"] = np.maximum(gs["near_surface_air_temperature"] - TBASE, 0.0)

print("Aggregating yearly features …")
feat = (
    gs.groupby(["scenario", "code_dep", "nom_dep", "year"], as_index=False)
      .agg(
          tmean_gs=("near_surface_air_temperature", "mean"),
          tmax_gs=("daily_maximum_near_surface_air_temperature", "mean"),
          prcp_gs=("precipitation_mm_day", "sum"),
          hot_days=("hot_day", "sum"),
          gdd=("gdd_day", "sum"),
          max_consec_dry=("dry_day", max_consecutive_ones),
      )
)
feat["dep_norm"] = feat["nom_dep"].map(norm_dep)

# ── train model ──────────────────────────────────────────────────────────
print("Training XGBoost model …")
feat_hist = feat[feat["scenario"] == "historical"].copy()
train_df = barley.merge(feat_hist, on=["dep_norm", "year"], how="inner")

feature_cols_base = ["tmean_gs", "tmax_gs", "prcp_gs", "hot_days", "gdd", "max_consec_dry"]
feature_cols_plus = feature_cols_base + ["year"]
target_col = "yield"

df_model = train_df.dropna(subset=feature_cols_base + [target_col]).copy()
tr = df_model[df_model["year"] < 2011].copy()

X_tr = pd.get_dummies(
    tr[feature_cols_plus + ["dep_norm"]],
    columns=["dep_norm"],
    drop_first=True,
)
y_tr = tr[target_col]

xgb2 = XGBRegressor(
    n_estimators=1500, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)
xgb2.fit(X_tr, y_tr)
print(f"  Training columns: {X_tr.shape[1]}")

# ── predict future (SSP scenarios) ──────────────────────────────────────
print("Generating future predictions …")
scenarios = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
future_feat = feat[feat["scenario"].isin(scenarios)].dropna(subset=feature_cols_base).copy()
future_feat["year"] = future_feat["year"].astype("Int64")

X_future = pd.get_dummies(
    future_feat[feature_cols_plus + ["dep_norm"]],
    columns=["dep_norm"],
    drop_first=True,
)
X_future = X_future.reindex(columns=X_tr.columns, fill_value=0)
future_feat["pred_yield"] = xgb2.predict(X_future)

# ── baseline yield (2004-2014 avg per dept) ─────────────────────────────
baseline = (
    barley[(barley["year"] >= 2004) & (barley["year"] <= 2014)]
    .groupby("dep_norm", as_index=False)["yield"]
    .mean()
    .rename(columns={"yield": "baseline_yield"})
)

# ── area 2014 (fall back to 2004-2014 avg if 2014 missing) ──────────────
area_2014 = (
    barley[barley["year"] == 2014]
    .groupby("dep_norm", as_index=False)["area"]
    .first()
    .rename(columns={"area": "area_2014"})
)
area_fallback = (
    barley[(barley["year"] >= 2004) & (barley["year"] <= 2014)]
    .groupby("dep_norm", as_index=False)["area"]
    .mean()
    .rename(columns={"area": "area_2014"})
)
area_2014 = (
    area_2014
    .set_index("dep_norm")
    .combine_first(area_fallback.set_index("dep_norm"))
    .reset_index()
)

# ── assemble output ─────────────────────────────────────────────────────
proj = future_feat[
    ["scenario", "code_dep", "nom_dep", "dep_norm", "year",
     "pred_yield", "gdd", "max_consec_dry"]
].copy()
proj = proj.merge(baseline, on="dep_norm", how="left")
proj = proj.merge(area_2014, on="dep_norm", how="left")
proj["pred_production"] = proj["pred_yield"] * proj["area_2014"]
proj["yield_change_pct"] = (
    (proj["pred_yield"] - proj["baseline_yield"]) / proj["baseline_yield"] * 100
)

proj.to_csv(out_path, index=False)
print(f"✅ Saved {len(proj)} rows → {out_path}")
