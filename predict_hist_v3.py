# predict_hist_v3.py
import os
import re
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RepeatedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import sparse

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def save_model(obj, filename):
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", filename)
    joblib.dump(obj, path)
    print(f"\nSaved trained model to: {path}")


def normalize_business_area(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^A-Z0-9\-\/\s:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*\/\s*", "/", s)
    s = re.sub(r"\s*:\s*", ":", s)
    return s


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def parse_budget_request(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["BusinessAreaNorm"] = df["Business Area"].apply(normalize_business_area)

    df["ProjType3"] = df["BusinessAreaNorm"].str.slice(0, 3).fillna("")
    df["ProjType6"] = df["BusinessAreaNorm"].str.slice(0, 6).fillna("")
    df["Sector"] = df["ProjType3"]

    first_token = df["BusinessAreaNorm"].str.split("-", n=1, expand=True)
    df["ProjRoot"] = first_token[0].fillna("")

    df["Quarter"] = df["Quarter"].astype(str).str.upper().str.strip()
    df.loc[~df["Quarter"].isin(["Q1", "Q2", "Q3", "Q4"]), "Quarter"] = "Q1"

    df["Fiscal Year"] = pd.to_numeric(df["Fiscal Year"], errors="coerce").fillna(0).astype(int)
    df["Region"] = pd.to_numeric(df["Region"], errors="coerce").fillna(0).astype(int)

    if "ClimateScore" not in df.columns:
        df["ClimateScore"] = 0.0
    df["ClimateScore"] = df["ClimateScore"].apply(lambda v: safe_float(v, 0.0))
    df["ClimateScore"] = np.clip(df["ClimateScore"], 0.0, 1.0)

    if "ClimateFlag" not in df.columns:
        df["ClimateFlag"] = (df["ClimateScore"] > 0).astype(int)
    df["ClimateFlag"] = pd.to_numeric(df["ClimateFlag"], errors="coerce").fillna(0).astype(int)

    if "GenderFlag" not in df.columns:
        df["GenderFlag"] = 0
    df["GenderFlag"] = pd.to_numeric(df["GenderFlag"], errors="coerce").fillna(0).astype(int)

    df["BA_Text"] = (
        df["BusinessAreaNorm"]
        + " |TYPE3=" + df["ProjType3"]
        + " |TYPE6=" + df["ProjType6"]
        + " |ROOT=" + df["ProjRoot"]
        + " |SECTOR=" + df["Sector"]
    )

    return df


def smape(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0


def wape(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(eps, np.sum(np.abs(y_true)))
    return (np.sum(np.abs(y_true - y_pred)) / denom) * 100.0


def densify_if_sparse(X):
    return X.toarray() if sparse.issparse(X) else X


def build_preprocessor():
    numeric_features = ["Region", "ClimateScore", "ClimateFlag", "GenderFlag"]
    categorical_features = ["Quarter", "ProjType3", "ProjType6", "ProjRoot", "Sector"]
    text_feature = "BA_Text"

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    text_transformer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_features=25000
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("txt", text_transformer, text_feature),
        ],
        remainder="drop",
        sparse_threshold=0.9
    )
    return preprocessor


def print_fold_example(df_feat: pd.DataFrame, y_ratio: np.ndarray, train_idx, test_idx, model_name: str):
    test_df = df_feat.iloc[test_idx].copy()

    print("\n" + "=" * 70)
    print(f"{model_name}: Example of CV Test Set (Fold 1)")
    print("=" * 70)
    print(f"Test rows: {len(test_df)} | Train rows: {len(train_idx)}")

    q_counts = test_df["Quarter"].value_counts().to_dict() if "Quarter" in test_df.columns else {}
    r_counts = test_df["Region"].value_counts().to_dict() if "Region" in test_df.columns else {}
    print(f"Test distribution - Quarter counts: {q_counts}")
    print(f"Test distribution - Region counts:  {r_counts}")

    y_test = y_ratio[test_idx]
    print(
        "Quarter-ratio target in test fold: "
        f"min={float(np.min(y_test)):.6f} median={float(np.median(y_test)):.6f} max={float(np.max(y_test)):.6f}"
    )

    cols_show = ["Fiscal Year", "Quarter", "Region", "Business Area", "ClimateFlag", "GenderFlag", "Budget Request"]
    cols_show = [c for c in cols_show if c in test_df.columns]
    tmp = test_df[cols_show].head(10).copy()
    tmp["y_qratio"] = y_test[: len(tmp)]
    print("\nSample test rows:")
    print(tmp.to_string(index=False))


def allocate_with_constraints(areas, preds, climate_flags, gender_flags, total_budget, min_climate=0.50, min_gender=0.40):
    n = len(areas)
    preds = np.asarray(preds, dtype=float)
    preds = np.clip(preds, 0, None)

    base = np.ones(n) / n if preds.sum() <= 0 else preds / preds.sum()
    alloc = base * float(total_budget)

    def shares(a):
        total = float(np.sum(a)) if np.sum(a) > 0 else 1.0
        c = float(np.sum(a[np.array(climate_flags, dtype=int) == 1])) / total
        g = float(np.sum(a[np.array(gender_flags, dtype=int) == 1])) / total
        return c, g

    max_iter = 8000
    step = 0.005
    alloc = alloc.astype(float)

    for _ in range(max_iter):
        c_share, g_share = shares(alloc)
        if c_share >= min_climate and g_share >= min_gender:
            break

        boost_mask = np.zeros(n, dtype=bool)
        if c_share < min_climate:
            boost_mask |= (np.array(climate_flags, dtype=int) == 1)
        if g_share < min_gender:
            boost_mask |= (np.array(gender_flags, dtype=int) == 1)

        reduce_mask = ~boost_mask
        if boost_mask.sum() == 0 or reduce_mask.sum() == 0:
            break

        give = step * total_budget
        take_each = give / reduce_mask.sum()

        alloc[reduce_mask] = np.maximum(0.0, alloc[reduce_mask] - take_each)
        alloc[boost_mask] += give / boost_mask.sum()

        total_now = float(np.sum(alloc))
        if total_now > 0:
            alloc *= (total_budget / total_now)

    return alloc


def _mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return float(np.mean(vals)), float(np.std(vals))


def _cv_baselines_and_model(pipe, X, df_feat_all, y_ratio, cv):
    maes_ml, wapes_ml, smapes_ml, r2s_ml = [], [], [], []
    maes_equal, wapes_equal, smapes_equal, r2s_equal = [], [], [], []
    maes_ba, wapes_ba, smapes_ba, r2s_ba = [], [], [], []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_ratio[train_idx], y_ratio[test_idx]

        pipe.fit(X_train, np.log1p(y_train))
        pred_ml = np.expm1(pipe.predict(X_test))
        pred_ml = np.clip(pred_ml, 0, None)

        maes_ml.append(mean_absolute_error(y_test, pred_ml))
        wapes_ml.append(wape(y_test, pred_ml))
        smapes_ml.append(smape(y_test, pred_ml))
        r2s_ml.append(r2_score(y_test, pred_ml))

        test_meta = df_feat_all.iloc[test_idx][["Fiscal Year", "Quarter"]]
        counts = test_meta.groupby(["Fiscal Year", "Quarter"]).size().to_dict()
        pred_equal = np.array([1.0 / counts[(fy, q)] for fy, q in test_meta.to_records(index=False)], dtype=float)

        maes_equal.append(mean_absolute_error(y_test, pred_equal))
        wapes_equal.append(wape(y_test, pred_equal))
        smapes_equal.append(smape(y_test, pred_equal))
        r2s_equal.append(r2_score(y_test, pred_equal))

        ba_mean = (
            df_feat_all.iloc[train_idx]
            .groupby("Business Area")["y_ratio"]
            .mean()
            .to_dict()
        )
        global_mean = float(np.mean(y_train))
        test_ba = df_feat_all.iloc[test_idx]["Business Area"].tolist()
        pred_ba = np.array([ba_mean.get(ba, global_mean) for ba in test_ba], dtype=float)

        maes_ba.append(mean_absolute_error(y_test, pred_ba))
        wapes_ba.append(wape(y_test, pred_ba))
        smapes_ba.append(smape(y_test, pred_ba))
        r2s_ba.append(r2_score(y_test, pred_ba))

    return {
        "ML": {
            "MAE_pp": _mean_std(np.array(maes_ml) * 100.0),
            "WAPE": _mean_std(wapes_ml),
            "sMAPE": _mean_std(smapes_ml),
            "R2": _mean_std(r2s_ml),
        },
        "EqualSplit": {
            "MAE_pp": _mean_std(np.array(maes_equal) * 100.0),
            "WAPE": _mean_std(wapes_equal),
            "sMAPE": _mean_std(smapes_equal),
            "R2": _mean_std(r2s_equal),
        },
        "BA_mean": {
            "MAE_pp": _mean_std(np.array(maes_ba) * 100.0),
            "WAPE": _mean_std(wapes_ba),
            "sMAPE": _mean_std(smapes_ba),
            "R2": _mean_std(r2s_ba),
        },
        "folds": len(maes_ml),
    }


def _cv_blend(pipe, X, df_feat_all, y_ratio, cv, alpha_grid):
    alpha_mae = {a: [] for a in alpha_grid}

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_ratio[train_idx], y_ratio[test_idx]

        ba_mean = (
            df_feat_all.iloc[train_idx]
            .groupby("Business Area")["y_ratio"]
            .mean()
            .to_dict()
        )
        global_mean = float(np.mean(y_train))
        test_ba = df_feat_all.iloc[test_idx]["Business Area"].tolist()
        pred_ba = np.array([ba_mean.get(ba, global_mean) for ba in test_ba], dtype=float)

        pipe.fit(X_train, np.log1p(y_train))
        pred_ml = np.expm1(pipe.predict(X_test))
        pred_ml = np.clip(pred_ml, 0, None)

        for a in alpha_grid:
            pred_bl = a * pred_ml + (1.0 - a) * pred_ba
            alpha_mae[a].append(mean_absolute_error(y_test, pred_bl))

    alpha_means = {a: float(np.mean(v)) for a, v in alpha_mae.items()}
    best_alpha = min(alpha_means, key=alpha_means.get)

    maes, wapes, smapes, r2s = [], [], [], []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_ratio[train_idx], y_ratio[test_idx]

        ba_mean = (
            df_feat_all.iloc[train_idx]
            .groupby("Business Area")["y_ratio"]
            .mean()
            .to_dict()
        )
        global_mean = float(np.mean(y_train))
        test_ba = df_feat_all.iloc[test_idx]["Business Area"].tolist()
        pred_ba = np.array([ba_mean.get(ba, global_mean) for ba in test_ba], dtype=float)

        pipe.fit(X_train, np.log1p(y_train))
        pred_ml = np.expm1(pipe.predict(X_test))
        pred_ml = np.clip(pred_ml, 0, None)

        pred_bl = best_alpha * pred_ml + (1.0 - best_alpha) * pred_ba

        maes.append(mean_absolute_error(y_test, pred_bl))
        wapes.append(wape(y_test, pred_bl))
        smapes.append(smape(y_test, pred_bl))
        r2s.append(r2_score(y_test, pred_bl))

    return {
        "best_alpha": float(best_alpha),
        "alpha_means": alpha_means,
        "Blend": {
            "MAE_pp": _mean_std(np.array(maes) * 100.0),
            "WAPE": _mean_std(wapes),
            "sMAPE": _mean_std(smapes),
            "R2": _mean_std(r2s),
        },
        "folds": len(maes),
    }


def main():
    print("Loading data...")
    df = pd.read_csv("data.csv", sep=";")
    print(f"Loaded {len(df)} records\n")

    for c in ["Fiscal Year", "Quarter", "Region", "Business Area", "Budget Request"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["Budget Request"] = parse_budget_request(df["Budget Request"])
    df = df[df["Budget Request"].notna() & (df["Budget Request"] > 0)].copy()
    df["Budget Request"] = df["Budget Request"].astype(float)

    df_feat = add_engineered_features(df)

    print(f"Unique regions: {sorted(df_feat['Region'].unique().tolist())}")
    print(f"Unique business areas: {df_feat['BusinessAreaNorm'].nunique()}\n")

    df_feat["QuarterSum"] = df_feat.groupby(["Fiscal Year", "Quarter"])["Budget Request"].transform("sum")
    df_feat["y_ratio"] = (df_feat["Budget Request"] / df_feat["QuarterSum"]).astype(float)
    y_ratio = np.clip(df_feat["y_ratio"].values.astype(float), 0, None)
    y_log = np.log1p(y_ratio)

    print("=" * 70)
    print("TARGET CHECK (Quarter Ratio)")
    print("=" * 70)
    print(f"y_ratio summary: min={np.min(y_ratio):.6f} median={np.median(y_ratio):.6f} max={np.max(y_ratio):.6f}")
    rs = df_feat.groupby(["Fiscal Year", "Quarter"]).apply(lambda g: float((g["Budget Request"] / g["Budget Request"].sum()).sum()))
    print("Example quarter ratio sums (should be ~1.0 each):")
    for k, v in rs.head(8).items():
        print(f"  {k[0]} {k[1]}: {v:.6f}")
    print()

    X = df_feat[[
        "Quarter", "Region", "Business Area",
        "ClimateScore", "ClimateFlag", "GenderFlag",
        "BusinessAreaNorm", "ProjType3", "ProjType6", "ProjRoot", "Sector", "BA_Text"
    ]].copy()

    preprocessor = build_preprocessor()

    model = HistGradientBoostingRegressor(
        random_state=42,
        loss="squared_error"
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("dense", FunctionTransformer(densify_if_sparse, accept_sparse=True)),
        ("model", model)
    ])

    param_dist = {
        "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
        "model__max_depth": [None, 6, 10, 14],
        "model__max_leaf_nodes": [31, 63, 127, 255],
        "model__min_samples_leaf": [10, 20, 40, 80],
        "model__l2_regularization": [0.0, 0.1, 0.5, 1.0, 3.0],
        "model__early_stopping": [False],
    }

    rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)

    fold1_train_idx, fold1_test_idx = next(rkf.split(X))
    print_fold_example(df_feat, y_ratio, fold1_train_idx, fold1_test_idx, "HistGradientBoostingRegressor (v3)")

    print("\nTUNING: RandomizedSearchCV")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=120,
        scoring="neg_mean_absolute_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X, y_log)
    best_pipe = search.best_estimator_

    alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    blend_summary = _cv_blend(best_pipe, X, df_feat[["Fiscal Year", "Quarter", "Business Area", "y_ratio"]].copy(), y_ratio, rkf, alpha_grid)

    baseline_summary = _cv_baselines_and_model(
        best_pipe,
        X,
        df_feat[["Fiscal Year", "Quarter", "Business Area", "y_ratio"]].copy(),
        y_ratio,
        rkf
    )

    print("\n" + "=" * 78)
    print("BASELINE VS MODEL COMPARISON (Quarter Ratio, Cross-Validated)")
    print("=" * 78)
    print(
        f"ML (HistGB tuned)          | MAE(pp) {baseline_summary['ML']['MAE_pp'][0]:6.2f} (±{baseline_summary['ML']['MAE_pp'][1]:.2f}) "
        f"| WAPE {baseline_summary['ML']['WAPE'][0]:6.2f}% | sMAPE {baseline_summary['ML']['sMAPE'][0]:6.2f}% | R² {baseline_summary['ML']['R2'][0]:7.4f}"
    )
    print(
        f"Baseline: EqualSplit       | MAE(pp) {baseline_summary['EqualSplit']['MAE_pp'][0]:6.2f} (±{baseline_summary['EqualSplit']['MAE_pp'][1]:.2f}) "
        f"| WAPE {baseline_summary['EqualSplit']['WAPE'][0]:6.2f}% | sMAPE {baseline_summary['EqualSplit']['sMAPE'][0]:6.2f}% | R² {baseline_summary['EqualSplit']['R2'][0]:7.4f}"
    )
    print(
        f"Baseline: BA mean          | MAE(pp) {baseline_summary['BA_mean']['MAE_pp'][0]:6.2f} (±{baseline_summary['BA_mean']['MAE_pp'][1]:.2f}) "
        f"| WAPE {baseline_summary['BA_mean']['WAPE'][0]:6.2f}% | sMAPE {baseline_summary['BA_mean']['sMAPE'][0]:6.2f}% | R² {baseline_summary['BA_mean']['R2'][0]:7.4f}"
    )
    print(
        f"Blend (alpha*ML+(1-a)*BA)  | MAE(pp) {blend_summary['Blend']['MAE_pp'][0]:6.2f} (±{blend_summary['Blend']['MAE_pp'][1]:.2f}) "
        f"| WAPE {blend_summary['Blend']['WAPE'][0]:6.2f}% | sMAPE {blend_summary['Blend']['sMAPE'][0]:6.2f}% | R² {blend_summary['Blend']['R2'][0]:7.4f}"
    )
    print(f"\nBest alpha: {blend_summary['best_alpha']:.2f}")
    print()

    print("\n" + "=" * 70)
    print("POST-TRAIN TESTS (after fitting on full dataset)")
    print("=" * 70)

    best_pipe.fit(X, y_log)

    ba_mean_full = df_feat.groupby("Business Area")["y_ratio"].mean().to_dict()
    global_mean_full = float(df_feat["y_ratio"].mean())
    best_alpha = float(blend_summary["best_alpha"])

    bundle = {
        "pipeline": best_pipe,
        "alpha": best_alpha,
        "ba_mean": ba_mean_full,
        "global_mean": global_mean_full
    }
    save_model(bundle, "ifc_budget_model_hist_v3_blend.joblib")

    example = {
        "Fiscal Year": 2025,
        "Quarter": "Q1",
        "Region": 1,
        "Business Area": "MAS-Health-Health Other",
        "ClimateScore": 0.0,
        "ClimateFlag": 0,
        "GenderFlag": 1,
        "Budget Request": 1.0
    }
    ex_df = pd.DataFrame([example])
    ex_df = add_engineered_features(ex_df)

    X_ex = ex_df[[
        "Quarter", "Region", "Business Area",
        "ClimateScore", "ClimateFlag", "GenderFlag",
        "BusinessAreaNorm", "ProjType3", "ProjType6", "ProjRoot", "Sector", "BA_Text"
    ]]

    pred_ml = float(np.clip(np.expm1(best_pipe.predict(X_ex))[0], 0, None))
    pred_ba = float(ba_mean_full.get(example["Business Area"], global_mean_full))
    pred_blend = float(best_alpha * pred_ml + (1.0 - best_alpha) * pred_ba)

    print("\nTEST: single estimate (quarter ratio)")
    print(f"ML predicted quarter ratio:      {pred_ml:.6f}")
    print(f"BA-mean predicted quarter ratio: {pred_ba:.6f}")
    print(f"BLEND predicted quarter ratio:   {pred_blend:.6f}")

    print("\nTEST: allocation with constraints (>=50% climate, >=40% gender) using BLEND")
    selected = [
        ("MAS-HEALTH-HEALTH OTHER", 0.0, 0, 1),
        ("MAS-MANUFACTURING-VALUE CHAIN FOR MANUFACTURING", 0.0, 0, 0),
        ("MAS-AGRIBUSINESS-CROP PRODUCTION", 0.7, 1, 0),
        ("FIG-SUSTAINABILITY AND CLIMATE-GREEN BUILDINGS", 0.9, 1, 1),
    ]

    rows = []
    for area, cscore, cflag, gflag in selected:
        rows.append({
            "Fiscal Year": 2025,
            "Quarter": "Q1",
            "Region": 1,
            "Business Area": area,
            "ClimateScore": float(cscore),
            "ClimateFlag": int(cflag),
            "GenderFlag": int(gflag),
            "Budget Request": 1.0
        })

    tdf = pd.DataFrame(rows)
    tdf = add_engineered_features(tdf)

    X_t = tdf[[
        "Quarter", "Region", "Business Area",
        "ClimateScore", "ClimateFlag", "GenderFlag",
        "BusinessAreaNorm", "ProjType3", "ProjType6", "ProjRoot", "Sector", "BA_Text"
    ]]

    pred_ml_t = np.clip(np.expm1(best_pipe.predict(X_t)), 0, None)
    bas = tdf["Business Area"].tolist()
    pred_ba_t = np.array([ba_mean_full.get(ba, global_mean_full) for ba in bas], dtype=float)
    pred_bl_t = best_alpha * pred_ml_t + (1.0 - best_alpha) * pred_ba_t

    areas = tdf["BusinessAreaNorm"].tolist()
    climate_flags = tdf["ClimateFlag"].tolist()
    gender_flags = tdf["GenderFlag"].tolist()

    total_budget = 1_000_000
    alloc = allocate_with_constraints(areas, pred_bl_t, climate_flags, gender_flags, total_budget, 0.50, 0.40)

    print(f"{'Area':<65} {'Pred(Blend)':>15} {'Alloc':>12} {'Climate':>8} {'Gender':>8}")
    print("-" * 110)
    for i, area in enumerate(areas):
        print(
            f"{area[:65]:<65} "
            f"{pred_bl_t[i]:>14.6f} "
            f"${alloc[i]:>11,.0f} "
            f"{str(int(climate_flags[i])):>8} "
            f"{str(int(gender_flags[i])):>8}"
        )

    total_alloc = float(np.sum(alloc))
    c_share = float(np.sum(alloc[np.array(climate_flags) == 1]) / total_alloc) if total_alloc > 0 else 0.0
    g_share = float(np.sum(alloc[np.array(gender_flags) == 1]) / total_alloc) if total_alloc > 0 else 0.0

    print(f"\nTotal allocated: ${total_alloc:,.0f}")
    print(f"Climate share: {c_share*100:.2f}%")
    print(f"Gender share: {g_share*100:.2f}%")
    print(f"Constraint check: {'OK' if (c_share >= 0.50 and g_share >= 0.40) else 'FAIL'}")


if __name__ == "__main__":
    main()