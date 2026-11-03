"""Portable working copies; frozen results are never overwritten by notebooks."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("acs", "vascular")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_hash(value, old, new):
    if isinstance(value, dict):
        return {k: v if k == "original_data_sha256" else replace_hash(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_hash(v, old, new) for v in value]
    return new if value == old else value


def initialize(cohort):
    if cohort not in COHORTS:
        raise ValueError(cohort)
    out = ROOT / "runs" / cohort
    out.mkdir(parents=True, exist_ok=True)
    marker = out / ".public_export_initialized"
    if not marker.exists():
        # Stop if a half-initialized directory contains work: never silently replace it.
        if any(out.iterdir()):
            raise RuntimeError(f"Nonempty uninitialized work directory: {out}")
        for source in (ROOT / "results" / cohort).iterdir():
            if source.is_file():
                shutil.copy2(source, out / source.name)
        shutil.copy2(ROOT / "data" / f"{cohort}.csv", out / "01_данные.csv")
        shutil.copy2(ROOT / "metadata" / f"{cohort}.json", out / "01_конфигурация.json")
        shutil.copy2(ROOT / "data" / f"{cohort}_dictionary.csv", out / "01_словарь.csv")
        meta = json.loads((out / "01_конфигурация.json").read_text())
        old, new = meta["original_data_sha256"], meta["data_sha256"]
        for path in out.glob("*.json"):
            obj = json.loads(path.read_text())
            obj = replace_hash(obj, old, new)
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
        (out / "PUBLIC_EXPORT.json").write_text(json.dumps({
            "cohort": cohort, "original_data_sha256": old, "public_data_sha256": new,
            "note": "Working derivative of recorded results. Data hash changed after identifier removal; clinical features and labels unchanged. This initialization is not model retraining."
        }, ensure_ascii=False, indent=2) + "\n")
        marker.write_text("1\n")
    # Runtime notebook hashes refer to these portable copies.
    for path in (ROOT / "notebooks" / cohort).glob("*.ipynb"):
        destination = out / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    return out
