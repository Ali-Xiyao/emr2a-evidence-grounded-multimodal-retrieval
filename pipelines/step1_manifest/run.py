# Wrapper for pipeline entrypoint. Use build_manifest.py.
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "build_manifest.py"), run_name="__main__")
