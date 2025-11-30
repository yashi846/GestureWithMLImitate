"""
集約したデータセットを多層パーセプトロン(MLP)を用いてを学習する

やることの流れ
1) 集約済みCSVを読み込む
2) ラベルをダミー変数へ変換する
3) 学習データとテストデータに分割する
4) 特徴量を標準化する
5) MLPで学習する
6) テストデータで精度を評価する

モデルの活性化関数
- 隠れ層1: ReLU
- 隠れ層2: ReLU
- 出力:ソフトマックス

"""

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


AGGREGATED_CSV_PATH = Path(
    "DataSet/Aggregated/DataSet_aggregated_demo_remove_none_1s.csv"
)
FEATURES: Sequence[str] = [
    "wtDoppler",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
]

FRAME_SIZE: int = 15
NUM_CLASSES: int = 4  # CLASSESとはジェスチャーの種類を指す

# データ分割と学習のパラメータ
TEST_SIZE: float = 0.2  # 全体のデータに対するテストデータの比率
RANDOM_STATE: int = 42  # 乱数のSeed値
HIDDEN_LAYER_SIZES: Tuple[int, int] = (30, 60)  # 第1層と第2層のノード数
MAX_ITER: int = 300  # 学習の繰り返し回数の最大値
ITER_NO_CHANGE = 10  # Validation scoreがITER_NO_CHANGE回連続伸びなかったら学習を止める
EARLY_STOPPING: bool = True  # 検証スコアが改善されない場合に、トレーニングを終了する


def resolve_csv_path() -> Path:
    if AGGREGATED_CSV_PATH.exists:
        return AGGREGATED_CSV_PATH
    raise FileNotFoundError(f"候補ファイルが見つかりません: {AGGREGATED_CSV_PATH}")


def build_feature_columns(
    features: Sequence[str] = FEATURES, frames: int = FRAME_SIZE
) -> List[str]:
    return [f"{bf}_{i}" for i in range(frames) for bf in features]


def load_data(feature_cols: Sequence[str], csv_path: Path = AGGREGATED_CSV_PATH):
    df = pd.read_csv(csv_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"必要な特徴列が不足: {missing}\n利用可能列の例: {list(df.columns)[:20]} ..."
        )

    if "label" not in df.columns:
        raise ValueError("'label' 列が見つかりません")

    features = df[feature_cols].values
    label = df["label"].values

    if features.shape[1] != len(FEATURES) * FRAME_SIZE:
        print(
            f"[警告] 入力の次元数が想定と異なります (想定: {len(FEATURES) * FRAME_SIZE} 実際: {features[1]}). "
        )

    return features, label


def encode_labels(y):
    le = LabelEncoder()  # 質的変数を量的変数に変換するためクラス
    y_enc = le.fit_transform(y)  # 量的変数に変換した結果
    classes = list(
        le.classes_
    )  # 各クラスがどの量的変数に変換されたか e.g. [D2U, L2R, R2L]が返され、その順で0,1,2とダミー変数に変換される

    if len(classes) != NUM_CLASSES:
        print(
            f"[警告] クラス数が想定と異なります (想定: {NUM_CLASSES} 実際: {len(classes)}). "
        )

    return y_enc, classes


# ============ メイン処理 ============
def main():
    csv_path = resolve_csv_path()
    print(f"使用したCSV: {csv_path}")

    feature_cols = build_feature_columns()

    # 1) データ読み込み
    features, label = load_data(feature_cols=feature_cols)

    # 2) ラベルをダミー変数に変換する
    label_quanted, classes = encode_labels(label)

    print(f"特徴次元: {features.shape[1]} データ数: {features.shape[0]}")
    print(f"クラス一覧 ({len(classes)}): {classes}")
    print(
        f"第1層のノード数: {HIDDEN_LAYER_SIZES[0]} 第2層のノード数: {HIDDEN_LAYER_SIZES[1]}"
    )

    # 3) 学習データとテストデータに分割
    # stratifyはクラスの比率を学習データとテストデータで保つために指定している
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        label_quanted,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=label_quanted,
    )

    # 4) 特徴量の標準化
    scaler = StandardScaler()  # 標準化するためのクラス
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) MLP を構築して学習
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",  # 確率的勾配ベースの最適化
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=EARLY_STOPPING,
        n_iter_no_change=ITER_NO_CHANGE,
        verbose=False,  # 学習の進行を表示させるか否か
    )

    clf.fit(X_train_scaled, y_train)

    # 6) 評価
    print("\n=== 学習完了 ===")
    print(f"反復回数: {clf.n_iter_}")
    print(f"交差エントロピー誤差: {clf.loss_:.4f}")

    y_pred = clf.predict(X_test_scaled)  # テストデータを学習結果を用いて予測する

    print("\n=== Classification Report ===")
    # zero_divisionは、ゼロ除算があったときに返す値
    print(
        classification_report(
            y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0
        )
    )

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 9) モデル保存
    # from joblib import dump
    # dump({"model": clf, "scaler": scaler, "label_encoder": le}, "mlp_model.joblib")


if __name__ == "__main__":
    main()
