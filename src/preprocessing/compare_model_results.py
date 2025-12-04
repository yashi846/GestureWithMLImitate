"""
学習モデルの結果（Accuracy、Confusion Matrix）を比較するスクリプト
複数のモデルの精度と混同行列を視覚化する
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict

MODELS = {
    "MLP (30,60)": {
        "accuracy": 0.60,
        "confusion_matrix": np.array(
            [[15, 3, 3, 2], [9, 15, 1, 3], [4, 3, 13, 5], [3, 3, 1, 17]],
        ),
    },
    "Prototypical + MLP": {
        "accuracy": 0.68,
        "confusion_matrix": np.array(
            [
                [968, 219, 249, 64],
                [249, 1165, 50, 36],
                [351, 4, 798, 347],
                [43, 105, 179, 1173],
            ]
        ),
    },
    "SVM": {
        "accuracy": 0.69,
        "confusion_matrix": np.array(
            [[15, 5, 2, 1], [3, 22, 2, 1], [5, 0, 13, 7], [0, 2, 3, 19]]
        ),
    },
    "RandomForest (30F)": {
        "accuracy": 0.78,
        "confusion_matrix": np.array(
            [[9, 1, 0, 1], [0, 11, 0, 1], [2, 1, 7, 1], [1, 1, 1, 8]]
        ),
    },
    "Prototypical + LSTM (30F)": {
        "accuracy": 0.8640,
        "confusion_matrix": np.array(
            [[449, 47, 4, 0], [32, 463, 5, 0], [29, 39, 392, 40], [0, 55, 21, 424]]
        ),
    },
}

CLASS_NAMES = ["D2U", "L2R", "R2L", "U2D"]
IMAGE_PATH = Path("data/image")


def plot_accuracy_comparison(models: Dict) -> None:
    """モデルのAccuracyを比較する棒グラフを生成"""
    model_names = list(models.keys())
    accuracies = [models[name]["accuracy"] for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        model_names,
        accuracies,
        color=["salmon", "lightgreen", "gold", "plum"],
        edgecolor="black",
        alpha=0.7,
    )

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = IMAGE_PATH / Path("accuracy_comparison.png")
    plt.savefig(save_path, dpi=100)


def plot_confusion_matrices(models: Dict, class_names: List[str]) -> None:
    """複数のモデルの混同行列を1枚の画像に出力"""
    num_models = len(models)
    rows = 2
    cols = num_models // rows + 1

    # 横長レイアウトのため横幅をモデル数に応じて広げる
    plt.figure(figsize=(5 * cols, 5 * rows))

    for idx, (model_name, model_data) in enumerate(models.items(), start=1):
        plt.subplot(rows, cols, idx)

        cm = model_data["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True,
        )

        plt.xlabel("Predicted", fontsize=11)
        plt.ylabel("Actual", fontsize=11)
        plt.title(
            f"{model_name}\n(Accuracy: {model_data['accuracy']:.2%})",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    save_path = IMAGE_PATH / Path("confusion_matrices_comparison.png")
    plt.savefig(save_path, dpi=100)


def main():
    plot_accuracy_comparison(MODELS)
    plot_confusion_matrices(MODELS, CLASS_NAMES)
    print("\nAll visualizations completed!")


if __name__ == "__main__":
    main()
