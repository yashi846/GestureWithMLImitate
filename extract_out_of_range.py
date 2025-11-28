from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

# ========== 設定（ベースから実行する想定） ==========
BASE_DIR = Path(".")
INPUT_ROOT = BASE_DIR / "DataSet"
OUTPUT_ROOT = BASE_DIR / "DataSet/RemoveNone"

SUBFOLDERS = ("D2U", "L2R", "R2L", "U2D")
FILE_GLOB = "*.csv"
WT_DOPPLER_COL = "wtDoppler"

# 判定範囲（例: -1.0 ~ 1.0）。実行前に設定してください。
RANGE_MIN: float = -3.40221  # 例: -1.0
RANGE_MAX: float = 2.36933  # 例:  1.0
INCLUSIVE: bool = True  # True: [min, max] を除外, False: (min, max) を除外

# I/O
ENCODING = "utf-8-sig"
NEWLINE = ""

# ========== 実装 ==========
def _require_range() -> Tuple[float, float]:
    if RANGE_MIN is None or RANGE_MAX is None:
        raise ValueError("RANGE_MIN と RANGE_MAX を設定してください。")
    if RANGE_MIN > RANGE_MAX:
        raise ValueError("RANGE_MIN は RANGE_MAX 以下である必要があります。")
    return float(RANGE_MIN), float(RANGE_MAX)


def _in_range(value: float, lo: float, hi: float) -> bool:
    if INCLUSIVE:
        return lo <= value <= hi
    return lo < value < hi


def filter_csv(in_path: Path, out_path: Path) -> Tuple[int, int]:
    lo, hi = _require_range()

    with in_path.open("r", encoding=ENCODING, newline=NEWLINE) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or WT_DOPPLER_COL not in reader.fieldnames:
            raise ValueError(f"{in_path}: 必須列 '{WT_DOPPLER_COL}' が見つかりません。")

        kept_rows = []
        removed = 0
        for row in reader:
            try:
                v = float(row[WT_DOPPLER_COL])
            except (TypeError, ValueError):
                # 数値化できない場合は安全側で保持
                kept_rows.append(row)
                continue

            if _in_range(v, lo, hi):
                removed += 1
            else:
                kept_rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding=ENCODING, newline=NEWLINE) as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return removed, len(kept_rows)


def main() -> None:
    total_in = 0
    total_out = 0

    for sub in SUBFOLDERS:
        in_dir = INPUT_ROOT / sub
        out_dir = OUTPUT_ROOT / sub

        if not in_dir.exists():
            print(f"警告: 入力フォルダが存在しません: {in_dir}")
            continue

        for in_path in sorted(in_dir.glob(FILE_GLOB)):
            out_path = out_dir / in_path.name
            removed, kept = filter_csv(in_path, out_path)
            total_in += removed + kept
            total_out += kept
            rel_out = out_path.relative_to(BASE_DIR)
            print(f"{in_path.name}: 除外 {removed} 行, 出力 {kept} 行 -> {rel_out}")

    print(f"合計: 入力 {total_in} 行, 出力 {total_out} 行, 除外 {total_in - total_out} 行")


if __name__ == "__main__":
    main()
