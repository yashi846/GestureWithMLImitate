"""
指定したデータセット(csv)に対し、ドップラー平均が指定した範囲に含まれている行のみを抽出し、新たなデータセットを出力するスクリプト
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

INPUT_ROOT = "./DataSet"
OUTPUT_ROOT = "./DataSet/RemoveNone"

SUBFOLDERS = ("D2U", "L2R", "R2L", "U2D")
WT_DOPPLER_COL = "wtDoppler"

RANGE_MIN: float = -1.95933
RANGE_MAX: float = 0.926441

ENCODING = "utf-8"


def filter_csv(in_path: Path, out_path: Path) -> Tuple[int, int]:
    with in_path.open("r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or WT_DOPPLER_COL not in reader.fieldnames:
            raise ValueError(f"{in_path}: 特徴量 '{WT_DOPPLER_COL}' が見つかりません。")

        kept_rows = []
        removed = 0
        for row in reader:
            if RANGE_MIN <= float(row[WT_DOPPLER_COL]) <= RANGE_MAX:
                removed += 1
            else:
                kept_rows.append(row)

    with out_path.open("w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return removed, len(kept_rows)


def main() -> None:
    total_in = 0
    total_out = 0

    for sub in SUBFOLDERS:
        in_dir = INPUT_ROOT / Path(sub)
        out_dir = OUTPUT_ROOT / Path(sub)

        if not in_dir.exists():
            print(f"警告: 入力フォルダが存在しません: {in_dir}")
            continue

        for in_path in sorted(in_dir.glob("*.csv")):
            out_path = out_dir / in_path.name
            removed, kept = filter_csv(in_path, out_path)
            total_in += removed + kept
            total_out += kept

    print(
        f"合計: 入力 {total_in} 行, 出力 {total_out} 行, 除外 {total_in - total_out} 行"
    )


if __name__ == "__main__":
    main()
