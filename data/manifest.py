import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            records: List[Dict[str, Any]] = []
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {idx} in {path}: {e}") from e
                if not isinstance(item, dict):
                    raise ValueError(f"Manifest line {idx} in {path} is not a JSON object.")
                records.append(item)
            return records

        data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Manifest item {i} in {path} is not a JSON object.")
            return data
        raise ValueError(f"Unsupported manifest format in {path}: expected JSON list or JSONL.")
