"""Execute portable notebooks in isolated kernels; results go to runs/."""
from pathlib import Path
import argparse
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "clinical-mortality-mpl"))
import nbformat
from nbclient import NotebookClient
from repository_support import ROOT, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, choices=["acs", "vascular"])
    parser.add_argument("--stage", choices=["data", "models", "shap", "all"], default="data")
    parser.add_argument("--kernel", default="python3", help="Kernel installed in the active environment")
    args = parser.parse_args()
    out = initialize(args.cohort)
    stages = ["01", "02", "03"] if args.stage == "all" else [{"data": "01", "models": "02", "shap": "03"}[args.stage]]
    for stage in stages:
        source = next((ROOT / "notebooks" / args.cohort).glob(stage + "_*.ipynb"))
        notebook = nbformat.read(source, as_version=4)
        client = NotebookClient(notebook, timeout=None, kernel_name=args.kernel, resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, out / source.name)
        print("Completed:", out / source.name, flush=True)


if __name__ == "__main__":
    main()
