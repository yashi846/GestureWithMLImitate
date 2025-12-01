"""
SVMで学習を行うスクリプト
以下のパラメータを最適化する
- C: コストパラメータ
- γ: RBFカーネルのパラメータ
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import optuna


CSV_PATH = Path(
    "../../data/DataSet/Aggregated/DataSet_aggregated_demo_remove_none_1s.csv"
)
FEATURES = [
    "wtDoppler",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
]
FRAME_SIZE = 15
N_TRIALS = 100


def load_data():
    df = pd.read_csv(CSV_PATH)

    # 90次元の特徴量カラム名を作成
    feature_cols = [f"{feat}_{i}" for i in range(FRAME_SIZE) for feat in FEATURES]

    X = df[feature_cols].values
    y = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, le.classes_


def objective(trial):
    X, y, _ = load_data()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Kernel: RBF
    # C: 正則化の強さ
    svc_c = trial.suggest_float("C", 1e-3, 1e3, log=True)
    # γ: 境界線の複雑さ
    svc_gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

    clf = SVC(C=svc_c, gamma=svc_gamma, kernel="rbf", random_state=42)

    # 交差検証
    scores = cross_val_score(clf, X, y, n_jobs=-1, cv=5)

    return scores.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 40)
    print(f"ベストスコア (CV平均): {study.best_value:.4f}")
    print("ベストパラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 40 + "\n")

    best_params = study.best_params

    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(
        C=best_params["C"],
        gamma=best_params["gamma"],
        kernel="rbf",
        probability=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("===== Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("===== Confusion Matrix =====")
    print(confusion_matrix(y_test, y_pred))
