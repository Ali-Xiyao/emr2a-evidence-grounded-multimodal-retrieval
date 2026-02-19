import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "evaluate_retrieval.py"), run_name="__main__")
