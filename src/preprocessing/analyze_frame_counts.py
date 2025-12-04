"""
ExtractGesture配下の各CSVファイルのデータ行数の統計情報を取得するスクリプト
平均値と標準偏差を計算する
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DATASET_ROOT = Path("data/DataSet/ExtractGesture/1s")
FRAME_SIZE = [15, 20, 25, 30, 35, 40, 45, 50]
FIG_PATH = Path("data/image/frame_counts_histogram.png")


def count_csv_rows(file_path: Path) -> int:
    """CSVファイルの行数（ヘッダ除く）を数える"""
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def plot_histograms(frame_counts: list) -> None:
    """クラス別のヒストグラムを生成"""
    class_data = {
        "D2U": [],
        "L2R": [],
        "R2L": [],
        "U2D": [],
    }

    for class_dir in sorted(DATASET_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for csv_file in sorted(class_dir.glob("*.csv")):
            row_count = count_csv_rows(csv_file)
            if class_name in class_data:
                class_data[class_name].append(row_count)

    plt.figure(figsize=(14, 10))
    colors = {"D2U": "salmon", "L2R": "lightgreen", "R2L": "gold", "U2D": "plum"}
    class_names = ["D2U", "L2R", "R2L", "U2D"]

    plt.subplot(2, 3, 1)
    plt.hist(frame_counts, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
    plt.xlabel("Frames")
    plt.ylabel("Frequency")
    plt.title("Overall Frame Distribution")
    plt.grid(axis="y", alpha=0.3)

    for idx, class_name in enumerate(class_names, start=2):
        plt.subplot(2, 3, idx)
        plt.hist(
            class_data[class_name],
            bins=15,
            edgecolor="black",
            alpha=0.7,
            color=colors[class_name],
        )
        plt.xlabel("Frames")
        plt.ylabel("Count")
        plt.title(f"{class_name} - Frame Distribution")
        plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=100)
    print("\nHistogram saved: frame_counts_histogram.png")


def analyze_frame_counts() -> None:
    """全CSVファイルの行数を集計し、統計情報を表示"""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT が存在しません: {DATASET_ROOT}")

    frame_counts = []
    file_info = []

    for class_dir in sorted(DATASET_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        class_counts = []

        for csv_file in sorted(class_dir.glob("*.csv")):
            row_count = count_csv_rows(csv_file)
            class_counts.append(row_count)
            frame_counts.append(row_count)
            file_info.append((csv_file.name, row_count))

        if class_counts:
            class_mean = np.mean(class_counts)
            class_std = np.std(class_counts)
            class_min = np.min(class_counts)
            class_max = np.max(class_counts)
            print(
                f"{class_dir.name} 統計: 平均={class_mean:.2f}, "
                f"標準偏差={class_std:.2f}, 最小={class_min}, 最大={class_max}"
            )

    print("\n" + "=" * 60)
    print("全体統計")
    print("=" * 60)
    overall_mean = np.mean(frame_counts)
    overall_std = np.std(frame_counts)
    overall_min = np.min(frame_counts)
    overall_max = np.max(frame_counts)
    total_files = len(frame_counts)

    print(f"総ファイル数: {total_files}")
    print(f"平均行数: {overall_mean:.2f}")
    print(f"標準偏差: {overall_std:.2f}")
    print(f"最小行数: {overall_min}")
    print(f"最大行数: {overall_max}")

    for fs in FRAME_SIZE:
        print(f"\n※ FRAME_SIZE={fs} の場合")
        valid_chunks = sum(1 for count in frame_counts if count >= fs)
        total_chunks = sum(count // fs for count in frame_counts)
        print(f"FRAME_SIZE以上のファイル数: {valid_chunks}/{total_files}")
        print(f"生成可能なチャンク数: {total_chunks}")

    plot_histograms(frame_counts)


if __name__ == "__main__":
    analyze_frame_counts()
