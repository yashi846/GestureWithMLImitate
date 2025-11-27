import csv
import os
from pathlib import Path
from typing import Dict, List


# ==========================
# 設定（リファクタリングしやすい定数）
# ==========================

# 入力データルート
DATASET_ROOT = Path("DataSet")

# 15行を1行にまとめるチャンクサイズ
CHUNK_SIZE = 15

# 使用する項目（入力CSVに存在する前提の列名）
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

# ラベルの決め方：フォルダ名 -> 出力ラベル
# ここを必要に応じて数値ラベルに変更してください（例：{"D2U": 0, "U2D": 1, ...}）
LABEL_MAPPING: Dict[str, str] = {
    # 例: そのままフォルダ名をラベルに使う
    # 数値ラベルにしたい場合は値を整数や文字列に変更
    "D2U": "D2U",
    "U2D": "U2D",
    "L2R": "L2R",
    "R2L": "R2L",
    "Push": "Push",
    "Pull": "Pull",
    "Shine": "Shine",
    "CWT": "CWT",
    "CCWT": "CCWT",
}

# 出力ファイルパス
OUTPUT_FILE = Path("DataSet/Aggregated/DataSet_aggregated_demonstration.csv")


def list_class_dirs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"DATASET_ROOT が存在しません: {root}")
    return [p for p in root.iterdir() if p.is_dir() and p.name in LABEL_MAPPING]


def build_header() -> List[str]:
    header: List[str] = []
    for i in range(CHUNK_SIZE):
        for col in FEATURE_COLUMNS:
            header.append(f"{col}_{i}")
    header.append("label")
    return header


def read_csv_rows(file_path: Path) -> List[Dict[str, str]]:
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # 確認: 必須列が存在するか
    missing = [c for c in FEATURE_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"必須列が見つかりません: {missing} in {file_path}. 見つかった列: {reader.fieldnames}"
        )
    return rows


def chunk_rows(rows: List[Dict[str, str]], chunk_size: int) -> List[List[Dict[str, str]]]:
    chunks: List[List[Dict[str, str]]] = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
        # 端数は破棄（必要ならゼロ埋めなどの処理を追加可）
    return chunks


def flatten_chunk(chunk: List[Dict[str, str]], label_value: str) -> List[str]:
    out: List[str] = []
    for i, row in enumerate(chunk):
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

        for class_dir in list_class_dirs(DATASET_ROOT):
            label = LABEL_MAPPING.get(class_dir.name)
            # クラスフォルダ内のCSVを走査
            for fp in sorted(class_dir.glob("*.csv")):
                rows = read_csv_rows(fp)
                chunks = chunk_rows(rows, CHUNK_SIZE)
                for ch in chunks:
                    writer.writerow(flatten_chunk(ch, label))
                    total_rows += 1

    print(f"書き出し完了: {OUTPUT_FILE}  生成行数={total_rows}")


if __name__ == "__main__":
    aggregate()
