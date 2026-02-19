import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "build_embeddings.py"), run_name="__main__")
