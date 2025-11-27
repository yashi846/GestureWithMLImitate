"""
初心者向け: Scikit-learn で多層パーセプトロン(MLP)を学習する最小コード

やることの流れ
1) 集約済みCSV(15フレーム×6特徴=90次元)を読み込む
2) ラベルを数値に変換する
3) 学習データとテストデータに分割する
4) 特徴量を標準化する(学習の安定化のため)
5) 指定の構成のMLPで学習する
6) テストデータで精度を評価する

モデル構成(要件どおり)
- 入力: 90 次元 (6特徴 × 15フレーム)
- 隠れ層1: 30 (ReLU)
- 隠れ層2: 60 (ReLU)
- 出力: 多クラス分類(内部でソフトマックス相当)

注意:
- データに含まれる実際のクラス数が 10 未満でも動作しますが、
  仕様上 10 クラスを想定しているならデータ側を揃える必要があります。
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# =====================================
# 設定(ここを変えるだけで実験できます)
# =====================================

# 集約CSVの候補パス(上から順に探します)
AGGREGATED_CSV_CANDIDATES: List[Path] = [
    Path("DataSet/Aggregated/DataSet_aggregated_demonstration.csv"),
    Path("DataSet_aggregated.csv"),
]

# 6つの基本特徴名(フレーム番号 _i が 0..14 で付きます)
# ヘッダ例: wtDoppler_0, wtRange_0, ..., azDoppCorr_14
SELECTED_BASE_FEATURES: Sequence[str] = [
    "wtDoppler",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
]

# フレーム数(集約時の設定と一致させる)
FRAMES: int = 15

# 想定クラス数(データと異なる場合は警告のみ表示)
EXPECTED_NUM_CLASSES: int = 4

# データ分割と学習のパラメータ
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
HIDDEN_LAYER_SIZES: Tuple[int, int] = (30, 60)  # 層1=30, 層2=60
MAX_ITER: int = 300
EARLY_STOPPING: bool = True  # 検証スコアが伸びなければ早めに止める


# ============ ヘルパー関数 ============
def resolve_csv_path() -> Path:
    """存在する最初のCSVパスを返します。見つからなければエラー。"""
    for p in AGGREGATED_CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"候補ファイルが見つかりません: {AGGREGATED_CSV_CANDIDATES}")


def build_feature_columns(base_features: Sequence[str], frames: int) -> List[str]:
    """'wtDoppler_0' のような列名リストを作る(合計 6×15=90 個)。"""
    return [f"{bf}_{i}" for i in range(frames) for bf in base_features]


def load_data(csv_path: Path, feature_cols: Sequence[str]):
    """CSVを読み込み、特徴行列Xとラベルyを返す。列の存在もチェックする。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"集約CSVが存在しません: {csv_path}")

    df = pd.read_csv(csv_path)

    # 必須列が全てあるか確認
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"必要な特徴列が不足: {missing}\n利用可能列の例: {list(df.columns)[:20]} ..."
        )

    if "label" not in df.columns:
        raise ValueError("'label' 列が見つかりません")

    X = df[feature_cols].values  # (サンプル数, 90)
    y = df["label"].values
    return X, y, df


def encode_labels(y):
    """文字ラベルを 0..C-1 の整数に変換し、エンコーダとクラス名一覧も返す。"""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = list(le.classes_)
    return y_enc, le, classes


# ============ メイン処理 ============
def main():
    # 1) CSV パスを解決
    csv_path = resolve_csv_path()
    print(f"使用CSV: {csv_path}")

    # 2) 使う 90 個の列名を用意
    feature_cols = build_feature_columns(SELECTED_BASE_FEATURES, FRAMES)

    # 3) データ読み込み
    X, y, _ = load_data(csv_path, feature_cols)

    # 4) ラベルを数値化
    y_enc, le, classes = encode_labels(y)

    print(f"特徴次元: {X.shape[1]} (期待 90) 実際={X.shape}")
    print(f"クラス一覧 ({len(classes)}): {classes}")
    if len(classes) != EXPECTED_NUM_CLASSES:
        print(
            f"[警告] クラス数が期待値 {EXPECTED_NUM_CLASSES} と異なります (実際 {len(classes)}). データのクラス構成を確認してください。"
        )

    # 5) 学習/テスト分割 (クラス比を保つため stratify を使用)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )

    # 6) 特徴量の標準化 (平均0, 分散1) -> MLP の収束が安定
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7) MLP を構築して学習
    # - activation="relu": 隠れ層の活性化関数
    # - hidden_layer_sizes=(30,60): 2層のユニット数
    # - early_stopping=True: 検証スコアが伸びなければ打ち切り
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation="relu",
        solver="adam",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=EARLY_STOPPING,
        n_iter_no_change=10,
        verbose=True,
    )

    clf.fit(X_train_scaled, y_train)

    # 8) テストデータで評価
    print("\n=== 学習完了 ===")
    print(f"最終反復回数: {clf.n_iter_}")
    print(f"トレーニング損失: {clf.loss_:.4f}")

    y_pred = clf.predict(X_test_scaled)

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test, y_pred, target_names=[str(c) for c in classes], zero_division=0
        )
    )

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 9) モデル保存をしたい場合は以下を有効化
    # from joblib import dump
    # dump({"model": clf, "scaler": scaler, "label_encoder": le}, "mlp_model.joblib")


if __name__ == "__main__":
    main()
