"""
複数のデータセット(csv)を1つのデータセット(csv)にまとめるスクリプト
"""

import csv
from pathlib import Path
from typing import Dict, List


DATASET_ROOT = Path("data/DataSet/ExtractGesture/1s")
OUTPUT_FILE = Path(
    "data/DataSet/Aggregated/DA_30frames_remove_none_1s.csv"
)


FRAME_SIZE = 30
FEATURE_COLUMNS: List[str] = [
    "wtDoppler",
    # "wtDopplerPos",
    # "wtDopplerNeg",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
    # "wtAzimuthStd",
    # "wtdElevStd",
]


def list_class_dirs(root: Path = DATASET_ROOT) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"DATASET_ROOT が存在しません: {root}")
    return [
        p for p in root.iterdir() if p.is_dir()
    ]  # iterdir()でファイルとサブディレクトリを取得 is_dir()でディレクトリかどうかを判断


def build_header() -> List[str]:
    header: List[str] = []
    for i in range(FRAME_SIZE):
        for col in FEATURE_COLUMNS:
            header.append(f"{col}_{i}")
    header.append("label")
    return header


def read_csv_rows(file_path: Path) -> List[Dict[str, str]]:
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # ヘッダ付きcsvファイルをディレクトリで読み出す
        rows = list(reader)
    missing = [c for c in FEATURE_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"以下の特徴量が{file_path}で見つかりません: {missing}")
    return rows


# チャンクを1行にしている
def chunk_rows(
    rows: List[Dict[str, str]], chunk_size: int
) -> List[List[Dict[str, str]]]:
    chunks: List[List[Dict[str, str]]] = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks


# 1行にされたチャンクをまとめている
def flatten_chunk(chunk: List[Dict[str, str]], label_value: str) -> List[str]:
    out: List[str] = []
    for _, row in enumerate(chunk):
        for col in FEATURE_COLUMNS:
            out.append(str(row.get(col, "")))
    out.append(str(label_value))
    return out


def aggregate() -> None:
    header = build_header()
    total_rows = 0

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for class_dir in list_class_dirs():
            label = class_dir.name  # フォルダ名をラベルに
            for fp in sorted(class_dir.glob("*.csv")):
                rows = read_csv_rows(fp)
                chunks = chunk_rows(rows, FRAME_SIZE)
                for ch in chunks:
                    writer.writerow(flatten_chunk(ch, label))
                    total_rows += 1

    print(f"書き出し完了: {OUTPUT_FILE}  生成行数={total_rows}")


if __name__ == "__main__":
    aggregate()
