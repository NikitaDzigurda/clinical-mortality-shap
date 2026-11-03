"""Recompute recorded metrics and check public data, splits, notebooks and SHAP."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clinical-mortality-mpl"))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def verify(cohort):
    checks = []

    def check(name, value):
        if not bool(value):
            raise AssertionError(f"{cohort}: {name}")
        checks.append(name)

    p = ROOT / "results" / cohort
    meta = json.loads((ROOT / "metadata" / f"{cohort}.json").read_text())
    datafile = ROOT / "data" / f"{cohort}.csv"
    d = pd.read_csv(datafile)
    n, events, groups = (906, 43, 892) if cohort == "acs" else (548, 55, 511)
    check("public_data_sha256", hashlib.sha256(datafile.read_bytes()).hexdigest() == meta["data_sha256"])
    check("allowlisted_columns", list(d) == ["row_id", "group_id"] + meta["features"] + ["hospital_death"])
    check("numeric_only_data", all(pd.api.types.is_numeric_dtype(d[c]) for c in d))
    check("cohort_size_and_deaths", len(d) == n and d.hospital_death.sum() == events)
    check("groups", d.group_id.nunique() == groups)
    check("row_numbers", np.array_equal(d.row_id, np.arange(n)))
    check("binary_target", set(d.hospital_death) == {0, 1})
    check("identifiers_complete", d[["row_id", "group_id", "hospital_death"]].notna().all().all())
    prefix = "02_" if cohort == "acs" else "02_обновлённые_"
    cases = [(p / ("02_oof_predictions.csv" if cohort == "acs" else prefix + "oof.csv"), p / (prefix + "метрики_блоков.csv"))]
    if cohort == "acs":
        cases += [(p / "02_условные_oof.csv", p / "02_условные_метрики_блоков.csv"),
                  (p / "04_baseline_default_oof.csv", p / "04_baseline_default_fold_metrics.csv")]
    checked_folds = 0
    for predictions, metricfile in cases:
        o = pd.read_csv(predictions)
        recorded = pd.read_csv(metricfile)
        recorded_key = "branch" if "branch" in recorded else "family"
        merged = o.merge(d[["row_id", "group_id", "hospital_death"]], on="row_id", suffixes=("", "_data"), validate="many_to_one")
        check(predictions.name + ":labels", np.array_equal(merged.y_true, merged.hospital_death))
        check(predictions.name + ":groups", np.array_equal(merged.group_id, merged.group_id_data))
        check(predictions.name + ":unique", not o.duplicated(["family", "repeat", "row_id"]).any())
        check(predictions.name + ":coverage", o.groupby(["family", "repeat"]).size().eq(n).all())
        check(predictions.name + ":probabilities", np.isfinite(o.p_death).all() and o.p_death.between(0, 1).all())
        for (family, repeat, fold), part in o.groupby(["family", "repeat", "fold"]):
            ref = recorded[(recorded[recorded_key] == family) & (recorded.repeat == repeat) & (recorded.fold == fold)]
            if cohort == "acs" and family == "prevalence_baseline" and len(ref) == 0:
                # This control was saved as OOF predictions, not as fold-table rows.
                check(f"constant_control:{repeat}:{fold}", part.p_death.nunique() == 1)
                check(f"constant_control_AP:{repeat}:{fold}", np.isclose(average_precision_score(part.y_true, part.p_death), part.y_true.mean(), atol=1e-12, rtol=0))
                checked_folds += 1
                continue
            check(f"{predictions.name}:{family}:{repeat}:{fold}:reference", len(ref) == 1)
            ref = ref.iloc[0]
            y, prob = part.y_true.to_numpy(), part.p_death.to_numpy()
            tn, fp, fn, tp = confusion_matrix(y, prob >= part.threshold.to_numpy(), labels=[0, 1]).ravel()
            metrics = {"AP": average_precision_score(y, prob), "ROC_AUC": roc_auc_score(y, prob),
                       "Brier": np.mean((y - prob) ** 2), "sensitivity": tp / (tp + fn),
                       "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
                       "precision": tp / (tp + fp) if tp + fp else 0,
                       "TN": tn, "FP": fp, "FN": fn, "TP": tp}
            for metric, value in metrics.items():
                if metric in ref.index:
                    check(f"{predictions.name}:{family}:{repeat}:{fold}:{metric}", np.isclose(value, ref[metric], atol=1e-7 if metric == "Brier" else 1e-12, rtol=0))
            checked_folds += 1
    for filename in [prefix + "внешние_разбиения.csv", prefix + "внутренние_разбиения.csv"]:
        splits = pd.read_csv(p / filename)
        keys = ["repeat", "fold"] + (["inner_fold"] if "inner_fold" in splits else [])
        for key, part in splits.groupby(keys):
            sides = list(part.side.unique())
            check(f"{filename}:{key}:two_sides", len(sides) == 2)
            a, b = (part[part.side == side] for side in sides)
            check(f"{filename}:{key}:group_separation", not set(a.group_id) & set(b.group_id))
            check(f"{filename}:{key}:row_separation", not set(a.row_id) & set(b.row_id))
            check(f"{filename}:{key}:group_labels", np.array_equal(part.group_id.to_numpy(), d.set_index("row_id").loc[part.row_id, "group_id"].to_numpy()))
    shapfile = p / ("03_shap_values.npz" if cohort == "acs" else "03_обновлённые_SHAP.npz")
    with np.load(shapfile, allow_pickle=False) as z:
        phi, base = z["values"], z["base_values"]
        target = z["model_output"] if "model_output" in z else z["p_death"]
        check("shap_coverage", phi.shape[0] == n)
        error = float(np.max(np.abs(base + phi.sum(axis=1) - target)))
        check("shap_additivity", error < 1e-5)
    # Numeric model inputs contain no direct identifiers. Check model output, not just SHAP sums.
    import joblib
    art = joblib.load(p / ("02_final_model.joblib" if cohort == "acs" else prefix + "final_model.joblib"))
    x = d[art["features"]]
    if cohort == "acs":
        prediction = art["model"].predict_proba(art["preprocessor"].transform(x))[:, 1]
    else:
        pre = art["preprocessor"]
        numeric = pre["scaler"].transform(pre["num_imputer"].transform(x[pre["nums"]]))
        cat = pre["encoder"].transform(pre["cat_imputer"].transform(x[pre["cats"]]))
        flags = x[pre["columns"]].isna().to_numpy(float) if pre["indicators"] else np.zeros((n, 0))
        raw = art["model"].predict_proba(np.c_[numeric, cat, flags])[:, 1].astype(float)
        ages = np.where(x.age_years.isna(), -1, np.searchsorted([60, 75], x.age_years.to_numpy(), side="right"))
        rates = np.array([pre["inclusion_rates"].get(f"{int(v)}_{int(age)}", 1.) for v, age in zip(x.vascular_group, ages)])
        prediction = rates * raw / (1 - raw + rates * raw)
    with np.load(shapfile, allow_pickle=False) as z:
        check("reloaded_model_prediction", np.allclose(prediction, z["p_death"], atol=1e-7, rtol=0))
    for path in (ROOT / "notebooks" / cohort).glob("*.ipynb"):
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                check(f"{path.name}:{i}:no_output", cell.get("outputs") == [] and cell.get("execution_count") is None)
                ast.parse("".join(cell["source"]))
    return {"cohort": cohort, "checks_passed": len(checks), "fold_records_recomputed": checked_folds,
            "rows": n, "death_marks": events, "groups": groups, "shap_max_additivity_error": error,
            "scope": "Saved artifacts, model prediction, public tables and portable syntax. No new Optuna tuning and no external clinical validation.",
            "checks": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["acs", "vascular", "all"], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [verify(c) for c in (["acs", "vascular"] if args.cohort == "all" else [args.cohort])]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    for report in reports:
        print(json.dumps({k: v for k, v in report.items() if k != "checks"}, ensure_ascii=False))
