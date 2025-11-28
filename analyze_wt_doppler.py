from __future__ import annotations
import sys
import csv
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 定数（要件: ファイルパス/ファイル名は定数で）
# =========================
DATA_DIR = Path("./DataSet/None")
OUTPUT_DIR = Path("./DataSet/Aggregated")
OUTPUT_FIG_PATH =  Path(f"{OUTPUT_DIR}/wtDoppler_hist.png")
SUPPORTED_EXTS = (".json", ".csv", ".npz", ".npy")
HIST_BINS = "fd"  # Freedman–Diaconis estimator
FIG_SIZE = (8, 5)
FIG_DPI = 120

# 大文字小文字や '_' の差異を吸収したキー判定用
WTD_KEYS_CANON = {"wtdoppler", "wt_doppler"}

def _canon(s: str) -> str:
    return s.lower().replace("_", "")

def _load_csv_wtd(path: Path) -> np.ndarray | None:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        # wtDoppler 列を特定
        tgt = None
        for col in reader.fieldnames:
            if _canon(col) == "wtdoppler":
                tgt = col
                break
        if tgt is None:
            return None
        vals: List[float] = []
        for row in reader:
            raw = row.get(tgt)
            if raw is None or raw.strip() == "":
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                # "1,2,3" や "1 2 3" 形への簡易対応
                parts = raw.replace(",", " ").split()
                for p in parts:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        pass
        if not vals:
            return None
        return np.asarray(vals, dtype=float)

def collect_all_wtd_values(dir_path: Path) -> np.ndarray:
    if not dir_path.exists():
        print(f"入力ディレクトリが存在しません: {dir_path}", file=sys.stderr)
        return np.empty(0, dtype=float)
    files = sorted(p for p in dir_path.rglob("*.csv"))
    arrays: List[np.ndarray] = []
    for f in files:
        arr = _load_csv_wtd(f)
        if arr is None or arr.size == 0:
            continue
        arrays.append(arr[np.isfinite(arr)])
    if not arrays:
        print("wtDoppler データが見つかりませんでした。", file=sys.stderr)
        return np.empty(0, dtype=float)
    return np.concatenate(arrays)

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values = collect_all_wtd_values(DATA_DIR)
    n = values.size
    if n == 0:
        return 1
    mean = float(np.mean(values))
    std_unbiased = float(np.std(values, ddof=1)) if n > 1 else float("nan")

    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    plt.hist(values, bins=HIST_BINS, edgecolor="black", alpha=0.8)
    plt.title("wtDoppler Histogram (All CSV in DataSet/None)")
    plt.xlabel("wtDoppler")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PATH)
    plt.close()

    print(f"対象ディレクトリ: {DATA_DIR}")
    print(f"データ点数: {n}")
    print(f"平均: {mean:.6g}")
    print(f"不偏標準偏差: {std_unbiased:.6g}")
    print(f"約95%の範囲: {mean-std_unbiased*2:.6g} ~ {mean+std_unbiased*2:.6g}")
    print(f"ヒストグラム保存先: {OUTPUT_FIG_PATH}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
