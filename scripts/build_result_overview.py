"""Build an overview from recorded numeric tables; supplied screenshots stay unchanged."""
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clinical-mortality-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    acs = pd.read_csv(ROOT / "results/acs/04_baseline_default_comparison_summary.csv")
    acs["cohort"] = "acs"
    v = pd.read_csv(ROOT / "results/vascular/02_обновлённые_сводка_блоков.csv")
    families = ["LR", "RF", "CatBoost", "XGBoost"]
    v = v[v.family.isin(families + [f + "_baseline" for f in families])].copy()
    v["variant"] = np.where(v.family.str.endswith("_baseline"), "default", "tuned")
    v["family"] = v.family.str.replace("_baseline", "", regex=False)
    v["cohort"] = "vascular"
    table = pd.concat([acs, v], ignore_index=True)[["cohort", "family", "variant", "AP", "ROC_AUC", "Brier", "precision"]]
    table.to_csv(ROOT / "docs/provenance/quality_factual.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (cohort, title) in enumerate([("acs", "а) ОКС: расчётные значения"), ("vascular", "б) Сосудистая задача")]):
        for row, metric in enumerate(["AP", "ROC_AUC"]):
            ax = axes[row, col]
            x = np.arange(4)
            for offset, variant, color, label in [(-.18, "default", "#9ba3aa", "Базовая модель"), (.18, "tuned", "#35677f", "Настройка Optuna")]:
                a = table[(table.cohort == cohort) & (table.variant == variant)].set_index("family").loc[families, metric]
                bars = ax.bar(x + offset, a, width=.36, color=color, label=label)
                ax.bar_label(bars, labels=[f"{z:.3f}".replace(".", ",") for z in a], fontsize=8, padding=3)
            ax.set_xticks(x, families)
            ax.set_ylabel(metric.replace("_", "-"))
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
            if row == 0:
                data = pd.read_csv(ROOT / "data" / f"{cohort}.csv").set_index("row_id")
                pref = "02_" if cohort == "acs" else "02_обновлённые_"
                splits = pd.read_csv(ROOT / "results" / cohort / (pref + "внешние_разбиения.csv"))
                prevalence = np.mean([data.loc[g.row_id, "hospital_death"].mean() for _, g in splits[splits.side == "test"].groupby(["repeat", "fold"])])
                ax.axhline(prevalence, color="grey", linestyle=":")
                ax.set_title(title)
                ax.set_ylim(0, 1 if cohort == "acs" else .22)
            else:
                ax.set_ylim(0, 1.06 if cohort == "acs" else .75)
                if cohort == "acs":
                    ax.set_yticks(np.linspace(0, 1, 6))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(ROOT / "docs/figures/01_quality_factual.png", dpi=170)
    plt.close(fig)
    cases = json.loads((ROOT / "docs/provenance/local_examples.json").read_text())
    local_checks = []
    for case, cohort, npz in zip(cases, ["acs", "vascular"], ["03_shap_values.npz", "03_обновлённые_SHAP.npz"]):
        with np.load(ROOT / "results" / cohort / npz, allow_pickle=False) as z:
            i = int(case["array_index"])
            assert np.isclose(z["p_death"][i], case["prediction"], atol=1e-12)
            assert np.isclose(z["base_values"][i], case["base"], atol=1e-12)
            assert np.isclose(case["base"] + sum(case["contributions"].values()), case["prediction"], atol=1e-12)
        local_checks.append({"cohort": cohort, "example_row": i, "matches_saved_model_and_shap": True})
    global_checks = []
    expected = {"acs": {"killip": .0277, "missing_crp": .0249, "crp": .0057, "age": .0054, "af": .0052, "pci": .0047, "ckd": .0031, "st_elevation": .0028},
                "vascular": {"vascular_group": .0281, "ldl_mmol_l": .0112, "crp_value": .0058, "bmi": .0041, "total_cholesterol": .0025, "age_years": .0015, "creatinine_value": .0015, "egfr_value": .0009}}
    for cohort, values in expected.items():
        file = "03_важности_SHAP.csv" if cohort == "acs" else "03_обновлённые_важности_SHAP.csv"
        a = pd.read_csv(ROOT / "results" / cohort / file).set_index("key")
        for key, val in values.items():
            assert abs(a.loc[key, "mean_abs_SHAP"] - val) <= .00005
        global_checks.append({"cohort": cohort, "screenshot_labels_checked": len(values), "tolerance_due_to_rounding": .00005})
    (ROOT / "verification/figure_check.json").write_text(json.dumps({"local": local_checks, "global": global_checks, "quality": "12 manually changed values in ACS screenshot; see quality_demo_changes.csv"}, ensure_ascii=False, indent=2) + "\n")
    print("Factual overview and screenshot consistency checks complete")


if __name__ == "__main__":
    main()
