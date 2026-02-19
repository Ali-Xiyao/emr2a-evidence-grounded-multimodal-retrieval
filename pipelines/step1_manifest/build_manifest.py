# build_manifest.py
# 示例：
#   python build_manifest.py --data_root data --out_dir data/processed
#
# 输入：
#   data_root/
#     Case_csv/*.csv                 （按类别的病例信息表）
#     CT_2D_image/<class>/<patient>/ （患者切片文件夹）
#
# 输出：
#   out_dir/manifest.jsonl           （每行一个病例 JSON）
#   out_dir/patient_mapping.csv      （仅在 CSV 无 patient_id 列时生成）
#   out_dir/missing_in_fs.csv        （CSV 有但图像文件夹缺失）
#   out_dir/missing_in_csv.csv       （图像文件夹有但 CSV 缺失）

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from pypinyin import lazy_pinyin as _lazy_pinyin  # type: ignore
    PINYIN_AVAILABLE = True
except Exception:
    _lazy_pinyin = None
    PINYIN_AVAILABLE = False

LABEL_DIR_MAP = {
    "Bacterial": "Bacterial",
    "Viral": "Viral",
    "PJP": "PJP",
    "Normal": "Normal",
}

CSV_LABEL_MAP = {
    "Bacterial.csv": "Bacterial",
    "Viral.csv": "Viral",
    "PJP.csv": "PJP",
    "Normal.csv": "Normal",
}

NAME_GUESS = ["name", "姓名", "患者姓名", "PatientName"]


def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def read_csv_with_fallback(path: Path, header: Optional[int] = "infer") -> pd.DataFrame:
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                header=header,
                dtype=str,
                keep_default_na=False,
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Failed to decode {path}")


def resolve_name_col(df: pd.DataFrame, name_col: Optional[str]) -> str:
    if name_col:
        if name_col not in df.columns:
            raise ValueError(f"Name column '{name_col}' not found. Columns: {list(df.columns)}")
        return name_col
    for guess in NAME_GUESS:
        if guess in df.columns:
            return guess
    raise ValueError(f"Name column not found. Columns: {list(df.columns)}")


def load_csvs(
    data_root: Path,
    name_col: Optional[str],
    id_col: str,
    label_col: Optional[str],
) -> Tuple[pd.DataFrame, str, bool]:
    rows: List[pd.DataFrame] = []
    csv_dir = data_root / "Case_csv"
    for csv_name, label in CSV_LABEL_MAP.items():
        csv_path = csv_dir / csv_name
        df = read_csv_with_fallback(csv_path)
        if label_col:
            if label_col not in df.columns:
                raise ValueError(
                    f"Label column '{label_col}' not found in {csv_name}. Columns: {list(df.columns)}"
                )
            df["__label__"] = df[label_col].astype(str)
        else:
            df["__label__"] = label
        rows.append(df)

    merged = pd.concat(rows, ignore_index=True)
    try:
        resolved_name_col = resolve_name_col(merged, name_col)
    except ValueError:
        # 若 CSV 无表头，尝试按无表头重新读取，并将第 1 列设为姓名列。
        rows = []
        for csv_name, label in CSV_LABEL_MAP.items():
            csv_path = csv_dir / csv_name
            df = read_csv_with_fallback(csv_path, header=None)
            if df.shape[1] == 5:
                df.columns = ["name", "sex", "age", "fever", "symptom"]
            else:
                cols = ["name"] + [f"col_{i}" for i in range(1, df.shape[1])]
                df.columns = cols
            if label_col:
                if label_col not in df.columns:
                    raise ValueError(
                        f"Label column '{label_col}' not found in {csv_name}. Columns: {list(df.columns)}"
                    )
                df["__label__"] = df[label_col].astype(str)
            else:
                df["__label__"] = label
            rows.append(df)
        merged = pd.concat(rows, ignore_index=True)
        resolved_name_col = "name"
        print("检测到 CSV 无表头，已按 [name/sex/age/fever/symptom] 自动设置列名。")
    has_id_col = id_col in merged.columns
    return merged, resolved_name_col, has_id_col


def parse_slice_exts(raw: str) -> Optional[List[str]]:
    if raw.strip() == "":
        return None
    exts = []
    for ext in raw.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.append(ext)
    return exts or None


def normalize_folder_key(value: str) -> str:
    value = value.strip().lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"__+", "_", value)
    return value.strip("_")


def normalize_person_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", value)
    return value


def name_to_pinyin(name: str) -> Optional[str]:
    if not PINYIN_AVAILABLE or _lazy_pinyin is None:
        return None
    pinyin = "_".join(_lazy_pinyin(name))
    return normalize_folder_key(pinyin)


def scan_images(
    data_root: Path,
    slice_exts: Optional[List[str]],
    image_subdir: str,
    relative_paths: bool,
    relative_base: Path,
) -> Dict[str, Dict[str, List[str]]]:
    image_root = data_root / image_subdir
    label_to_patients: Dict[str, Dict[str, List[str]]] = {}
    for label, dir_name in LABEL_DIR_MAP.items():
        category_dir = image_root / dir_name
        patients: Dict[str, List[str]] = {}
        if not category_dir.exists():
            label_to_patients[label] = patients
            continue
        # 每个 patient_dir 对应一个患者的切片文件夹。
        for patient_dir in sorted([p for p in category_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            files = []
            for item in patient_dir.iterdir():
                if not item.is_file():
                    continue
                if slice_exts and item.suffix.lower() not in slice_exts:
                    continue
                files.append(item)
            files_sorted = sorted(files, key=lambda p: natural_key(p.name))
            if relative_paths:
                patients[patient_dir.name] = [
                    str(p.relative_to(relative_base)) for p in files_sorted
                ]
            else:
                patients[patient_dir.name] = [str(p) for p in files_sorted]
        label_to_patients[label] = patients
    return label_to_patients


def sha1_id(name: str) -> str:
    return "P" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def align(
    df: pd.DataFrame,
    name_col: str,
    id_col: str,
    has_id_col: bool,
    label_col: Optional[str],
    image_index: Dict[str, Dict[str, List[str]]],
    drop_name: bool,
    image_subdir: str,
) -> Tuple[List[dict], Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    manifest: List[dict] = []
    missing_in_fs = []
    matched_folders: Dict[str, set] = {label: set() for label in LABEL_DIR_MAP}
    mapping_rows = []

    # 用 patient_id 或 name 将 CSV 记录对齐到患者文件夹。
    # 规范化图像目录中的患者文件夹名（去掉首尾空格）
    image_index = {
        label: {k.strip(): v for k, v in patients.items()}
        for label, patients in image_index.items()
    }
    normalized_index = {
        label: {normalize_person_name(k): k for k in patients.keys()}
        for label, patients in image_index.items()
    }

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        name_norm = normalize_person_name(name)
        if not name or name == "nan":
            continue

        label = str(row.get(label_col, row.get("__label__", ""))).strip()
        if label not in image_index:
            continue

        patient_id = None
        match_by = None
        folder_name = None
        label_patients = image_index[label]
        label_patients_norm = normalized_index[label]

        if has_id_col:
            raw_id = row.get(id_col)
            if pd.notna(raw_id):
                candidate = str(raw_id).strip()
                if candidate in label_patients:
                    patient_id = candidate
                    match_by = "patient_id"
                    folder_name = candidate
        if patient_id is None and name in label_patients:
            patient_id = str(row.get(id_col)).strip() if has_id_col and pd.notna(row.get(id_col)) else sha1_id(name)
            match_by = "name"
            folder_name = name
        if patient_id is None and name_norm in label_patients_norm:
            patient_id = str(row.get(id_col)).strip() if has_id_col and pd.notna(row.get(id_col)) else sha1_id(name)
            match_by = "name"
            folder_name = label_patients_norm[name_norm]
        if patient_id is None:
            pinyin_key = name_to_pinyin(name)
            if pinyin_key and pinyin_key in label_patients_norm:
                patient_id = str(row.get(id_col)).strip() if has_id_col and pd.notna(row.get(id_col)) else sha1_id(name)
                match_by = "name"
                folder_name = label_patients_norm[pinyin_key]

        # 若未找到匹配文件夹，记录失败并跳过。
        if patient_id is None:
            if not has_id_col:
                patient_id = sha1_id(name)
                match_by = "name"
            else:
                patient_id = str(row.get(id_col)).strip() if pd.notna(row.get(id_col)) else sha1_id(name)
                match_by = "name"
            missing_in_fs.append(
                {
                    "patient_id": patient_id,
                    "name": name,
                    "label": label,
                    "match_by": match_by,
                    "expected_dir": str(Path(image_subdir) / LABEL_DIR_MAP[label] / name),
                }
            )
            if not has_id_col:
                mapping_rows.append(
                    {
                        "patient_id": patient_id,
                        "name": name,
                        "label": label,
                        "folder_name": "",
                        "match_by": match_by,
                    }
                )
            continue

        slices = label_patients.get(folder_name, [])
        matched_folders[label].add(folder_name)
        meta = row.to_dict()
        for key in [name_col, id_col, "__label__"]:
            meta.pop(key, None)
        if label_col:
            meta.pop(label_col, None)

        entry = {
            "patient_id": patient_id if patient_id else sha1_id(name),
            "label": label,
            "image_dir": str(Path(image_subdir) / LABEL_DIR_MAP[label] / folder_name),
            "slices": slices,
            "meta": meta,
            "match_by": match_by or "name",
        }
        if not drop_name:
            entry["name"] = name

        manifest.append(entry)

        if not has_id_col:
            mapping_rows.append(
                {
                    "patient_id": entry["patient_id"],
                    "name": name,
                    "label": label,
                    "folder_name": folder_name,
                    "match_by": match_by or "name",
                }
            )

    missing_in_csv = []
    for label, patients in image_index.items():
        for folder_name in patients.keys():
            if folder_name not in matched_folders[label]:
                missing_in_csv.append(
                    {
                        "label": label,
                        "folder_name": folder_name,
                        "image_dir": str(Path(image_subdir) / LABEL_DIR_MAP[label] / folder_name),
                    }
                )

    mapping_df = pd.DataFrame(mapping_rows) if mapping_rows else None
    return manifest, mapping_df, pd.DataFrame(missing_in_fs), pd.DataFrame(missing_in_csv)


def write_manifest(out_dir: Path, manifest: List[dict]) -> None:
    out_path = out_dir / "manifest.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest from CSV + CT image folders.")
    parser.add_argument("--data_root", default="data", help="Dataset root directory.")
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", default="outputs", help="Output directory.")
    parser.add_argument("--name_col", default=None, help="Name column (auto-guess if not set).")
    parser.add_argument("--id_col", default="patient_id", help="Patient id column name.")
    parser.add_argument("--label_col", default=None, help="Optional label column in CSV.")
    parser.add_argument(
        "--image_subdir",
        default="CT_2D_image_core",
        help="Image subdir under data_root (e.g., CT_2D_image_core or CT_2D_image).",
    )
    parser.add_argument(
        "--relative_paths",
        default="true",
        choices=["true", "false"],
        help="Store slice paths relative to project root (data_root parent).",
    )
    parser.add_argument("--slice_ext", default=".png,.jpg,.jpeg,.dcm", help="Comma-separated extensions; empty=all.")
    parser.add_argument("--drop_name", action="store_true", help="Drop patient name in manifest.")
    args = parser.parse_args()

    if not PINYIN_AVAILABLE:
        print("未安装 pypinyin，姓名->拼音匹配不可用。如需匹配拼音文件夹，请先安装：pip install pypinyin")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, resolved_name_col, has_id_col = load_csvs(
        data_root=data_root,
        name_col=args.name_col,
        id_col=args.id_col,
        label_col=args.label_col,
    )

    slice_exts = parse_slice_exts(args.slice_ext)
    relative_paths = args.relative_paths == "true"
    image_index = scan_images(
        data_root=data_root,
        slice_exts=slice_exts,
        image_subdir=args.image_subdir,
        relative_paths=relative_paths,
        relative_base=data_root.parent,
    )

    manifest, mapping_df, missing_in_fs, missing_in_csv = align(
        df=df,
        name_col=resolved_name_col,
        id_col=args.id_col,
        has_id_col=has_id_col,
        label_col=args.label_col,
        image_index=image_index,
        drop_name=args.drop_name,
        image_subdir=args.image_subdir,
    )

    write_manifest(out_dir, manifest)

    if mapping_df is not None:
        mapping_df.to_csv(out_dir / "patient_mapping.csv", index=False, encoding="utf-8-sig")

    print(f"Manifest rows: {len(manifest)}")
    if manifest:
        print("Per-label counts:")
        for label, group in pd.DataFrame(manifest).groupby("label"):
            print(f"  {label}: {len(group)} patients")
    if not missing_in_fs.empty:
        missing_in_fs.to_csv(out_dir / "missing_in_fs.csv", index=False, encoding="utf-8-sig")
    if not missing_in_csv.empty:
        missing_in_csv.to_csv(out_dir / "missing_in_csv.csv", index=False, encoding="utf-8-sig")

    print(f"Alignment failures (CSV -> FS missing): {len(missing_in_fs)}")
    print(f"Alignment failures (FS -> CSV missing): {len(missing_in_csv)}")


if __name__ == "__main__":
    main()
