"""Exercise actual notebook training functions on one saved split, without Optuna."""
from pathlib import Path
import ast
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clinical-mortality-mpl"))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = []
    for cohort in ["acs", "vascular"]:
        meta = json.loads((ROOT / "metadata" / f"{cohort}.json").read_text())
        data = pd.read_csv(ROOT / "data" / f"{cohort}.csv")
        p = ROOT / "results" / cohort
        prefix = "02_" if cohort == "acs" else "02_обновлённые_"
        protocol = json.loads((p / (prefix + "протокол.json")).read_text())
        splits = pd.read_csv(p / (prefix + "внешние_разбиения.csv"))
        s = splits[(splits.repeat == 0) & (splits.fold == 0)]
        train_ids = s.loc[s.side == "train", "row_id"].to_numpy()
        test_ids = s.loc[s.side == "test", "row_id"].to_numpy()
        assert not set(data.group_id.iloc[train_ids]) & set(data.group_id.iloc[test_ids])
        notebook = json.loads((ROOT / "notebooks" / cohort / "02_модели_Optuna_и_эксперименты.ipynb").read_text())
        source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        imports = [n for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        out = ROOT / "runs" / "training_smoke" / cohort
        out.mkdir(parents=True, exist_ok=True)
        env = {"__name__": "training_smoke", "PROJECT_ROOT": ROOT, "OUT": out, "D": data,
               "meta": meta, "FEATURES": meta["features"], "RUN_ID": "portable_training_smoke",
               "BASE_SEED": 2026, "CONFIG": {}, "SAMPLING_AUDIT": [], "STRATUM_AUDIT": [],
               "PRE_CACHE": {}, "FIT_COUNT": 0}
        exec(compile(ast.Module(body=imports, type_ignores=[]), "notebook_imports", "exec"), env)
        env["savejson"] = lambda obj, name: (out / name).write_text(json.dumps(obj, default=str))
        definition_tree = ast.parse("".join(notebook["cells"][4]["source"]))
        definitions = [n for n in definition_tree.body if isinstance(n, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook_training_definitions", "exec"), env)
        x = data[meta["features"]]
        y = data.hospital_death.to_numpy()
        strategies = ["all_real", "stratified_4to1"] if cohort == "acs" else ["all_real", "under_5"]
        for strategy in strategies:
            for family in ["LR", "RF", "CatBoost", "XGBoost"]:
                cfg = {"strategy": strategy, "indicators": True, "params": protocol["baseline"][family]}
                artifact = env["train"](x.iloc[train_ids], y[train_ids], family, cfg, 2026)
                pred = env["predict"](artifact, x.iloc[test_ids])
                if isinstance(pred, tuple):
                    pred = pred[0]
                assert np.asarray(pred).shape == (len(test_ids),)
                assert np.isfinite(pred).all() and np.all((pred >= 0) & (pred <= 1))
                report.append({"cohort": cohort, "family": family, "strategy": strategy,
                               "test_rows": len(test_ids), "status": "passed"})
                print(cohort, family, strategy, "passed", flush=True)
    target = ROOT / "verification/training_smoke_check.json"
    target.write_text(json.dumps({"fits": report, "scope": "16 fits using actual notebook functions on one saved grouped split; no Optuna search"}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
