# app.py
import os
import re
import io
import json
import uuid
import time
import joblib
import numpy as np
import pandas as pd
import webbrowser
import threading
import subprocess

from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from scipy import sparse  # IMPORTANT for densify_if_sparse

try:
    import sklearn  # noqa: F401
    import sklearn.pipeline  # noqa: F401
    import sklearn.compose  # noqa: F401
    import sklearn.preprocessing  # noqa: F401
    import sklearn.feature_extraction.text  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.linear_model  # noqa: F401
except Exception:
    pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

MODEL_PATH = os.path.join(BASE_DIR, "models", "ifc_budget_model_hist_v3_blend.joblib")
LOG_PATH = os.path.join(BASE_DIR, "investment_log.csv")
BUDGETS_PATH = os.path.join(BASE_DIR, "regional_budgets.json")

DEFAULT_FISCAL_YEAR = 2025
HOST = "127.0.0.1"
PORT = 5001

app = Flask(__name__, template_folder=TEMPLATES_DIR)

bundle = None
pipeline = None
alpha = 1.0
ba_mean = {}
global_mean = 0.0

investment_log = []  # list[dict]


def densify_if_sparse(X):
    return X.toarray() if sparse.issparse(X) else X


def normalize_business_area(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = s.replace("&", " AND ")
    s = re.sub(r"[\u2010-\u2015]", "-", s)                  # normalize unicode hyphens
    s = re.sub(r"[^A-Z0-9\-\/\s:]", " ", s)                # keep A-Z, 0-9, -, /, space, :
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-\s*", "-", s)                         # normalize around hyphen
    s = re.sub(r"\s*\/\s*", "/", s)                        # normalize around slash
    s = re.sub(r"\s*:\s*", ":", s)                         # normalize around colon
    return s


def fingerprint_for_search(s: str) -> str:
    n = normalize_business_area(s)
    return re.sub(r"[^A-Z0-9]", "", n)


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


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


def yn_to_flag(x: str) -> int:
    x = str(x).strip().lower()
    return 1 if x in {"yes", "y", "true", "1"} else 0


def yn_to_climate_score(x: str) -> float:
    return 1.0 if yn_to_flag(x) == 1 else 0.0


def parse_money(x: str) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    s = s.replace("$", "").replace(" ", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def money_2dp(x: float) -> float:
    try:
        return float(f"{float(x):.2f}")
    except Exception:
        return 0.0


def parse_region(region_text: str) -> int:
    s = str(region_text).strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return 1


def normalize_quarter(q: str) -> str:
    q = str(q or "Q1").strip().upper()
    return q if q in {"Q1", "Q2", "Q3", "Q4"} else "Q1"

def default_budgets():
    return {str(i): 0.0 for i in range(1, 6)}


def load_budgets():
    if os.path.exists(BUDGETS_PATH):
        try:
            with open(BUDGETS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = default_budgets()
            for i in range(1, 6):
                out[str(i)] = float(data.get(str(i), 0.0) or 0.0)
            return out
        except Exception:
            return default_budgets()
    return default_budgets()


def save_budgets(budgets: dict):
    try:
        with open(BUDGETS_PATH, "w", encoding="utf-8") as f:
            json.dump(budgets, f, indent=2)
    except Exception:
        pass


def sum_invested_by_region(log_rows):
    totals = {str(i): 0.0 for i in range(1, 6)}
    for r in log_rows:
        reg = str(int(r.get("Region", 1)))
        inv = float(r.get("Invested", 0.0) or 0.0)
        if reg in totals:
            totals[reg] += inv
    return totals


def compute_remaining(budgets: dict, log_rows):
    invested = sum_invested_by_region(log_rows)
    remaining = {}
    for i in range(1, 6):
        k = str(i)
        remaining[k] = float(budgets.get(k, 0.0) or 0.0) - float(invested.get(k, 0.0) or 0.0)
        remaining[k] = float(remaining[k])
    return remaining


def build_running_log_view(budgets: dict, log_rows):
    remaining_now = {str(i): float(budgets.get(str(i), 0.0) or 0.0) for i in range(1, 6)}
    view = []
    for r in log_rows:
        reg = str(int(r.get("Region", 1)))
        invested = float(r.get("Invested", 0.0) or 0.0)

        before = remaining_now.get(reg, 0.0)
        after = before - invested

        rr = dict(r)
        rr["Remaining Before"] = before
        rr["Remaining After"] = after
        view.append(rr)

        remaining_now[reg] = after

    return view

_ba_display_list = []
_ba_norm_set = set()
_ba_fp_set = set()
_ba_rows = []


def _build_ba_search_cache():
    global _ba_display_list, _ba_norm_set, _ba_fp_set, _ba_rows

    raw_keys = list((ba_mean or {}).keys())
    rows = []
    for k in raw_keys:
        disp = normalize_business_area(k)
        norm = disp
        fp = fingerprint_for_search(disp)
        if not norm:
            continue
        rows.append({"display": disp, "norm": norm, "fp": fp})

    by_norm = {}
    for r in rows:
        by_norm.setdefault(r["norm"], r)
    rows = list(by_norm.values())

    rows.sort(key=lambda r: r["display"])

    _ba_rows = rows
    _ba_display_list = [r["display"] for r in rows]
    _ba_norm_set = set(r["norm"] for r in rows)
    _ba_fp_set = set(r["fp"] for r in rows)


def load_model():
    global bundle, pipeline, alpha, ba_mean, global_mean

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find model file at: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Joblib file must contain a dict with at least: {'pipeline': ...}")

    pipeline = bundle["pipeline"]
    alpha = float(bundle.get("alpha", 1.0))
    ba_mean = bundle.get("ba_mean", {}) or {}
    global_mean = float(bundle.get("global_mean", 0.0))

    _build_ba_search_cache()


def load_log():
    global investment_log
    if os.path.exists(LOG_PATH):
        try:
            df = pd.read_csv(LOG_PATH)
            rows = df.to_dict(orient="records")
            for r in rows:
                if not r.get("id"):
                    r["id"] = uuid.uuid4().hex
                r["Region"] = int(parse_region(r.get("Region", 1)))
                r["Quarter"] = normalize_quarter(r.get("Quarter", "Q1"))
                r["Project Type"] = normalize_business_area(r.get("Project Type", ""))
                r["Climate"] = str(r.get("Climate", "No"))
                r["Gender"] = str(r.get("Gender", "No"))
                r["Invested"] = float(r.get("Invested", 0.0) or 0.0)
            investment_log = rows
        except Exception:
            investment_log = []
    else:
        investment_log = []


def save_log():
    try:
        df = pd.DataFrame(investment_log)
        cols = ["id", "Region", "Quarter", "Project Type", "Climate", "Gender", "Invested"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
        df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass

def predict_quarter_ratio(region: int, quarter: str, business_area: str, climate_yn: str, gender_yn: str) -> dict:
    if pipeline is None:
        raise RuntimeError("Pipeline not loaded. Did load_model() run?")

    ba_input = normalize_business_area(business_area)

    climate_score = yn_to_climate_score(climate_yn)
    climate_flag = 1 if climate_score > 0 else 0
    gender_flag = yn_to_flag(gender_yn)

    row = {
        "Fiscal Year": DEFAULT_FISCAL_YEAR,
        "Quarter": normalize_quarter(quarter),
        "Region": int(region),
        "Business Area": ba_input,
        "ClimateScore": float(climate_score),
        "ClimateFlag": int(climate_flag),
        "GenderFlag": int(gender_flag),
        "Budget Request": 1.0,
    }

    df = pd.DataFrame([row])
    df_feat = add_engineered_features(df)

    X = df_feat[
        [
            "Quarter",
            "Region",
            "Business Area",
            "ClimateScore",
            "ClimateFlag",
            "GenderFlag",
            "BusinessAreaNorm",
            "ProjType3",
            "ProjType6",
            "ProjRoot",
            "Sector",
            "BA_Text",
        ]
    ].copy()

    pred_ml = float(np.clip(np.expm1(pipeline.predict(X))[0], 0, None))
    pred_ba = float(ba_mean.get(ba_input, global_mean))
    pred_blend = float(alpha * pred_ml + (1.0 - alpha) * pred_ba)

    is_known = (ba_input in _ba_norm_set) or (fingerprint_for_search(ba_input) in _ba_fp_set)

    return {
        "business_area_norm": ba_input,
        "pred_ml": pred_ml,
        "pred_ba": pred_ba,
        "pred_blend": pred_blend,
        "is_known_type": bool(is_known),
    }

def safe_weights_from_invested(rows):
    inv = np.array([float(r.get("Invested", 0.0) or 0.0) for r in rows], dtype=float)
    total = float(inv.sum())
    if total <= 0:
        return None
    return inv / total


def distribute_amount_cents(total_amount: float, weights: np.ndarray):
    total_amount = max(0.0, float(total_amount))
    cents_total = int(round(total_amount * 100))

    raw = weights * cents_total
    base = np.floor(raw).astype(int)
    remainder = cents_total - int(base.sum())

    frac = raw - base
    order = np.argsort(-frac)
    for i in range(remainder):
        base[order[i % len(base)]] += 1

    out = [b / 100.0 for b in base.tolist()]
    return out


def split_amount_across_regions(total_amount: float, remaining_by_region: dict):
    total_amount = money_2dp(max(0.0, float(total_amount)))
    rem = {i: max(0.0, float(remaining_by_region.get(str(i), 0.0) or 0.0)) for i in range(1, 6)}
    total_rem = sum(rem.values())

    if total_amount <= 0 or total_rem <= 0:
        return {i: 0.0 for i in range(1, 6)}

    weights = np.array([rem[i] / total_rem for i in range(1, 6)], dtype=float)
    parts = distribute_amount_cents(total_amount, weights)
    return {i: money_2dp(parts[i - 1]) for i in range(1, 6)}
@app.route("/project_types", methods=["GET"])
def project_types():
    q_raw = request.args.get("q", "") or ""
    q_norm = normalize_business_area(q_raw)
    q_fp = fingerprint_for_search(q_raw)

    if not q_norm:
        return jsonify({"query": "", "matches": _ba_display_list[:50], "exact": False})

    exact = (q_norm in _ba_norm_set) or (q_fp in _ba_fp_set)

    starts_norm = [r["display"] for r in _ba_rows if r["norm"].startswith(q_norm)]
    starts_fp = [r["display"] for r in _ba_rows if (q_fp and r["fp"].startswith(q_fp) and not r["norm"].startswith(q_norm))]
    contains_norm = [r["display"] for r in _ba_rows if (q_norm in r["norm"] and not r["norm"].startswith(q_norm))]
    contains_fp = [r["display"] for r in _ba_rows if (q_fp and (q_fp in r["fp"]) and (q_norm not in r["norm"]) and (not r["fp"].startswith(q_fp)))]

    matches = []
    seen = set()
    for bucket in (starts_norm, starts_fp, contains_norm, contains_fp):
        for m in bucket:
            if m not in seen:
                matches.append(m)
                seen.add(m)
            if len(matches) >= 50:
                break
        if len(matches) >= 50:
            break

    return jsonify({"query": q_norm, "matches": matches, "exact": bool(exact)})


@app.route("/", methods=["GET"])
def index():
    budgets = load_budgets()
    remaining_by_region = compute_remaining(budgets, investment_log)
    log_view = build_running_log_view(budgets, investment_log)
    msg = request.args.get("msg", "")
    return render_template(
        "index.html",
        result=None,
        budgets=budgets,
        remaining_by_region=remaining_by_region,
        log=log_view[::-1][:25],
        msg=msg,
        dist_preview=None,
        climate_default="No",
        gender_default="No",
    )


@app.route("/set_budgets", methods=["POST"])
def set_budgets():
    budgets = default_budgets()
    for i in range(1, 6):
        budgets[str(i)] = money_2dp(parse_money(request.form.get(f"budget_{i}", "0")))
        if budgets[str(i)] < 0:
            budgets[str(i)] = 0.0

    save_budgets(budgets)
    return redirect(url_for("index", msg="Budgets updated."))


@app.route("/estimate", methods=["POST"])
def estimate():
    budgets = load_budgets()
    remaining_by_region = compute_remaining(budgets, investment_log)

    region_text = request.form.get("region", "")
    quarter = normalize_quarter(request.form.get("quarter", "Q1"))
    project_type = request.form.get("project_type", "")

    climate_yn = request.form.get("climate_yn", "No")
    gender_yn = request.form.get("gender_yn", "No")

    region = parse_region(region_text)
    remaining_val = float(remaining_by_region.get(str(region), 0.0) or 0.0)
    remaining_val_clamped = max(0.0, remaining_val)

    pred = predict_quarter_ratio(region, quarter, project_type, climate_yn, gender_yn)

    suggested = pred["pred_blend"] * remaining_val_clamped
    suggested = min(suggested, remaining_val_clamped)
    suggested = money_2dp(suggested)

    new_remaining = money_2dp(max(0.0, remaining_val_clamped - suggested))

    result = {
        "region": region,
        "quarter": quarter,
        "remaining_before": float(remaining_val_clamped),
        "project_type": pred["business_area_norm"],
        "project_type_known": bool(pred["is_known_type"]),
        "climate": climate_yn,
        "gender": gender_yn,
        "pred_quarter_ratio": float(pred["pred_blend"]),
        "recommended": float(suggested),
        "remaining_after": float(new_remaining),
    }

    log_view = build_running_log_view(budgets, investment_log)

    return render_template(
        "index.html",
        result=result,
        budgets=budgets,
        remaining_by_region=compute_remaining(budgets, investment_log),
        log=log_view[::-1][:25],
        msg="",
        dist_preview=None,
        climate_default=climate_yn,
        gender_default=gender_yn,
    )


@app.route("/confirm", methods=["POST"])
def confirm():
    budgets = load_budgets()

    region = int(parse_region(request.form.get("region", "1")))
    quarter = normalize_quarter(request.form.get("quarter", "Q1"))
    project_type = request.form.get("project_type", "").strip()

    climate = request.form.get("climate", "No")
    gender = request.form.get("gender", "No")

    invested = money_2dp(parse_money(request.form.get("invested", "0")))
    if invested < 0:
        invested = 0.0

    remaining_by_region = compute_remaining(budgets, investment_log)
    remaining_val = max(0.0, float(remaining_by_region.get(str(region), 0.0) or 0.0))
    if invested > remaining_val:
        invested = money_2dp(remaining_val)

    entry = {
        "id": uuid.uuid4().hex,
        "Region": int(region),
        "Quarter": quarter,
        "Project Type": normalize_business_area(project_type),
        "Climate": str(climate),
        "Gender": str(gender),
        "Invested": float(invested),
    }

    investment_log.append(entry)
    save_log()

    return redirect(url_for("index", msg="Project logged."))


@app.route("/update_entry", methods=["POST"])
def update_entry():
    budgets = load_budgets()
    entry_id = request.form.get("id", "").strip()
    new_invested = money_2dp(parse_money(request.form.get("invested", "0")))
    if new_invested < 0:
        new_invested = 0.0

    idx = None
    for i, r in enumerate(investment_log):
        if str(r.get("id", "")) == entry_id:
            idx = i
            break

    if idx is None:
        msg = "Could not find the selected log entry."
    else:
        region = int(parse_region(investment_log[idx].get("Region", 1)))

        remaining_by_region = compute_remaining(budgets, investment_log)
        remaining_now = max(0.0, float(remaining_by_region.get(str(region), 0.0) or 0.0))
        old_invested = float(investment_log[idx].get("Invested", 0.0) or 0.0)
        max_allowed = remaining_now + old_invested
        if new_invested > max_allowed:
            new_invested = money_2dp(max_allowed)

        investment_log[idx]["Invested"] = float(new_invested)
        save_log()
        msg = "Investment amount updated."

    return redirect(url_for("index", msg=msg))


@app.route("/delete_entry", methods=["POST"])
def delete_entry():
    entry_id = request.form.get("id", "").strip()
    before = len(investment_log)
    investment_log[:] = [r for r in investment_log if str(r.get("id", "")) != entry_id]
    after = len(investment_log)

    if after < before:
        save_log()
        msg = "Entry deleted."
    else:
        msg = "Could not find the selected entry."

    return redirect(url_for("index", msg=msg))

@app.route("/distribute_preview", methods=["POST"])
def distribute_preview():
    budgets = load_budgets()
    remaining_by_region = compute_remaining(budgets, investment_log)

    quarter = normalize_quarter(request.form.get("dist_quarter", "Q1"))
    region_sel = request.form.get("dist_region", "All").strip()
    amount_mode = request.form.get("dist_amount_mode", "remaining").strip()
    custom_amount = money_2dp(parse_money(request.form.get("dist_custom_amount", "0")))

    if region_sel.lower() == "all":
        eligible_by_region = {}
        for i in range(1, 6):
            eligible_by_region[i] = [
                r for r in investment_log
                if int(parse_region(r.get("Region", 1))) == i and normalize_quarter(r.get("Quarter", "Q1")) == quarter
            ]

        rem_clamped = {i: max(0.0, float(remaining_by_region.get(str(i), 0.0) or 0.0)) for i in range(1, 6)}
        total_remaining_all = sum(rem_clamped.values())

        if amount_mode == "remaining":
            per_region_amount = {i: money_2dp(rem_clamped[i]) for i in range(1, 6)}
            amount_to_dist_total = money_2dp(sum(per_region_amount.values()))
        else:
            amount_to_dist_total = min(max(0.0, float(custom_amount)), float(total_remaining_all))
            amount_to_dist_total = money_2dp(amount_to_dist_total)
            per_region_amount = split_amount_across_regions(amount_to_dist_total, remaining_by_region)

        if amount_to_dist_total <= 0:
            log_view = build_running_log_view(budgets, investment_log)
            return render_template(
                "index.html",
                result=None,
                budgets=budgets,
                remaining_by_region=remaining_by_region,
                log=log_view[::-1][:25],
                msg="Amount to distribute is $0 (no remaining budget available).",
                dist_preview=None,
                climate_default="No",
                gender_default="No",
            )

        preview_rows = []
        warnings = []

        for region in range(1, 6):
            amt_region = float(per_region_amount.get(region, 0.0) or 0.0)
            if amt_region <= 0:
                continue

            eligible = eligible_by_region[region]
            if not eligible:
                warnings.append(f"Region {region}: no eligible logged projects in {quarter} (skipped).")
                continue

            weights = safe_weights_from_invested(eligible)
            if weights is None:
                warnings.append(f"Region {region}: eligible projects have total invested = 0 (skipped).")
                continue

            add_amounts = distribute_amount_cents(amt_region, weights)

            for r, add in zip(eligible, add_amounts):
                cur = float(r.get("Invested", 0.0) or 0.0)
                new = money_2dp(cur + float(add))
                preview_rows.append(
                    {
                        "id": r.get("id"),
                        "Region": int(parse_region(r.get("Region", 1))),
                        "Quarter": normalize_quarter(r.get("Quarter", "Q1")),
                        "Project Type": r.get("Project Type", ""),
                        "Climate": r.get("Climate", ""),
                        "Gender": r.get("Gender", ""),
                        "Invested_Current": money_2dp(cur),
                        "Invested_Add": money_2dp(add),
                        "Invested_New": new,
                    }
                )

        log_view = build_running_log_view(budgets, investment_log)

        if not preview_rows:
            msg = "No eligible projects found for distribution across all regions."
            if warnings:
                msg += " " + " ".join(warnings)
            return render_template(
                "index.html",
                result=None,
                budgets=budgets,
                remaining_by_region=remaining_by_region,
                log=log_view[::-1][:25],
                msg=msg,
                dist_preview=None,
                climate_default="No",
                gender_default="No",
            )

        dist_preview = {
            "quarter": quarter,
            "region_sel": region_sel,
            "amount_mode": amount_mode,
            "custom_amount": float(custom_amount),
            "amount_to_dist": float(amount_to_dist_total),
            "rows": preview_rows,
        }

        msg = "Review the distribution preview and confirm to apply (regions kept separate)."
        if warnings:
            msg += " " + " ".join(warnings)

        return render_template(
            "index.html",
            result=None,
            budgets=budgets,
            remaining_by_region=remaining_by_region,
            log=log_view[::-1][:25],
            msg=msg,
            dist_preview=dist_preview,
            climate_default="No",
            gender_default="No",
        )

    region = int(parse_region(region_sel))
    eligible = [
        r for r in investment_log
        if int(parse_region(r.get("Region", 1))) == region and normalize_quarter(r.get("Quarter", "Q1")) == quarter
    ]
    total_remaining = max(0.0, float(remaining_by_region.get(str(region), 0.0) or 0.0))

    log_view = build_running_log_view(budgets, investment_log)

    if not eligible:
        return render_template(
            "index.html",
            result=None,
            budgets=budgets,
            remaining_by_region=remaining_by_region,
            log=log_view[::-1][:25],
            msg="No eligible logged projects for that quarter/region selection.",
            dist_preview=None,
            climate_default="No",
            gender_default="No",
        )

    weights = safe_weights_from_invested(eligible)
    if weights is None:
        return render_template(
            "index.html",
            result=None,
            budgets=budgets,
            remaining_by_region=remaining_by_region,
            log=log_view[::-1][:25],
            msg="Cannot distribute because eligible projects have total invested = 0.",
            dist_preview=None,
            climate_default="No",
            gender_default="No",
        )

    if amount_mode == "custom":
        amount_to_dist = min(max(0.0, float(custom_amount)), float(total_remaining))
    else:
        amount_to_dist = float(total_remaining)

    amount_to_dist = money_2dp(amount_to_dist)

    if amount_to_dist <= 0:
        return render_template(
            "index.html",
            result=None,
            budgets=budgets,
            remaining_by_region=remaining_by_region,
            log=log_view[::-1][:25],
            msg="Amount to distribute is $0 (no remaining budget available).",
            dist_preview=None,
            climate_default="No",
            gender_default="No",
        )

    add_amounts = distribute_amount_cents(amount_to_dist, weights)

    preview_rows = []
    for r, add in zip(eligible, add_amounts):
        cur = float(r.get("Invested", 0.0) or 0.0)
        new = money_2dp(cur + float(add))
        preview_rows.append(
            {
                "id": r.get("id"),
                "Region": int(parse_region(r.get("Region", 1))),
                "Quarter": normalize_quarter(r.get("Quarter", "Q1")),
                "Project Type": r.get("Project Type", ""),
                "Climate": r.get("Climate", ""),
                "Gender": r.get("Gender", ""),
                "Invested_Current": money_2dp(cur),
                "Invested_Add": money_2dp(add),
                "Invested_New": new,
            }
        )

    dist_preview = {
        "quarter": quarter,
        "region_sel": region_sel,
        "amount_mode": amount_mode,
        "custom_amount": float(custom_amount),
        "amount_to_dist": float(amount_to_dist),
        "rows": preview_rows,
    }

    return render_template(
        "index.html",
        result=None,
        budgets=budgets,
        remaining_by_region=remaining_by_region,
        log=log_view[::-1][:25],
        msg="Review the distribution preview and confirm to apply.",
        dist_preview=dist_preview,
        climate_default="No",
        gender_default="No",
    )


@app.route("/distribute_confirm", methods=["POST"])
def distribute_confirm():
    budgets = load_budgets()

    quarter = normalize_quarter(request.form.get("dist_quarter", "Q1"))
    region_sel = request.form.get("dist_region", "All").strip()
    amount_mode = request.form.get("dist_amount_mode", "remaining").strip()
    custom_amount = money_2dp(parse_money(request.form.get("dist_custom_amount", "0")))

    remaining_by_region = compute_remaining(budgets, investment_log)

    if region_sel.lower() == "all":
        rem_clamped = {i: max(0.0, float(remaining_by_region.get(str(i), 0.0) or 0.0)) for i in range(1, 6)}
        total_remaining_all = sum(rem_clamped.values())

        if amount_mode == "remaining":
            per_region_amount = {i: money_2dp(rem_clamped[i]) for i in range(1, 6)}
            amount_to_dist_total = money_2dp(sum(per_region_amount.values()))
        else:
            amount_to_dist_total = min(max(0.0, float(custom_amount)), float(total_remaining_all))
            amount_to_dist_total = money_2dp(amount_to_dist_total)
            per_region_amount = split_amount_across_regions(amount_to_dist_total, remaining_by_region)

        if amount_to_dist_total <= 0:
            return redirect(url_for("index", msg="Amount to distribute is $0."))

        warnings = []

        for region in range(1, 6):
            amt_region = float(per_region_amount.get(region, 0.0) or 0.0)
            if amt_region <= 0:
                continue

            eligible = [
                r for r in investment_log
                if int(parse_region(r.get("Region", 1))) == region and normalize_quarter(r.get("Quarter", "Q1")) == quarter
            ]

            if not eligible:
                warnings.append(f"Region {region}: no eligible logged projects in {quarter} (skipped).")
                continue

            weights = safe_weights_from_invested(eligible)
            if weights is None:
                warnings.append(f"Region {region}: eligible projects have total invested = 0 (skipped).")
                continue

            add_amounts = distribute_amount_cents(amt_region, weights)
            id_to_add = {str(r.get("id")): float(add) for r, add in zip(eligible, add_amounts)}

            for r in investment_log:
                rid = str(r.get("id"))
                if rid in id_to_add:
                    r["Invested"] = money_2dp(float(r.get("Invested", 0.0) or 0.0) + float(id_to_add[rid]))

        save_log()

        msg = f"Distributed ${amount_to_dist_total:,.2f} across eligible projects (regions kept separate)."
        if warnings:
            msg += " " + " ".join(warnings)

        return redirect(url_for("index", msg=msg))

    region = int(parse_region(region_sel))
    total_remaining = max(0.0, float(remaining_by_region.get(str(region), 0.0) or 0.0))

    eligible = [
        r for r in investment_log
        if int(parse_region(r.get("Region", 1))) == region and normalize_quarter(r.get("Quarter", "Q1")) == quarter
    ]

    if not eligible:
        return redirect(url_for("index", msg="No eligible projects found when confirming."))

    weights = safe_weights_from_invested(eligible)
    if weights is None:
        return redirect(url_for("index", msg="Cannot distribute because eligible projects have total invested = 0."))

    if amount_mode == "custom":
        amount_to_dist = min(max(0.0, float(custom_amount)), float(total_remaining))
    else:
        amount_to_dist = float(total_remaining)

    amount_to_dist = money_2dp(amount_to_dist)
    if amount_to_dist <= 0:
        return redirect(url_for("index", msg="Amount to distribute is $0."))

    add_amounts = distribute_amount_cents(amount_to_dist, weights)
    id_to_add = {str(r.get("id")): float(add) for r, add in zip(eligible, add_amounts)}

    for r in investment_log:
        rid = str(r.get("id"))
        if rid in id_to_add:
            r["Invested"] = money_2dp(float(r.get("Invested", 0.0) or 0.0) + float(id_to_add[rid]))

    save_log()
    return redirect(url_for("index", msg=f"Distributed ${amount_to_dist:,.2f} across eligible projects."))


@app.route("/download", methods=["GET"])
def download():
    df = pd.DataFrame(investment_log) if investment_log else pd.DataFrame(
        columns=["id", "Region", "Quarter", "Project Type", "Climate", "Gender", "Invested"]
    )

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="investment_log.csv",
    )


@app.route("/clear", methods=["POST"])
def clear():
    investment_log.clear()
    save_log()
    return redirect(url_for("index", msg="Log cleared."))


@app.route("/clear_all", methods=["POST"])
def clear_all():
    investment_log.clear()
    save_log()

    budgets = default_budgets()
    save_budgets(budgets)

    return redirect(url_for("index", msg="Log and budgets cleared."))

def _shutdown_server():
    """
    Try graceful Werkzeug shutdown, then hard-exit as fallback.
    """
    try:
        fn = request.environ.get("werkzeug.server.shutdown")
        if fn:
            fn()
            return
    except Exception:
        pass

    os._exit(0)


@app.route("/quit", methods=["POST"])
def quit_app():
    # Quit in a background thread so the response can return cleanly.
    def _do_quit():
        time.sleep(0.15)
        try:
            _shutdown_server()
        except Exception:
            os._exit(0)

    threading.Thread(target=_do_quit, daemon=True).start()
    return ("", 204)


def open_ui(url: str):
    if os.name == "nt":
        # Edge "app window" tends to allow window.close() to work more often than a normal tab.
        try:
            subprocess.Popen(
                ["msedge", f"--app={url}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return
        except Exception:
            pass

    # fallback
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    load_model()
    load_log()

    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    url = f"http://{HOST}:{PORT}"
    threading.Timer(0.6, lambda: open_ui(url)).start()
    app.run(host=HOST, port=PORT, debug=False)