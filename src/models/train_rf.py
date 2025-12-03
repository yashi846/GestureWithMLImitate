"""
Random Forestで学習を行うスクリプト
以下のパラメータを最適化する
- n_estimators : 森の中の木の数
- max_depth : 木の最大の深さ
- min_samples_split : 内部ノードを分割するために必要な最小サンプル数
- min_samples_leaf : 葉ノードに必要な最小サンプル数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import optuna


CSV_PATH = Path("data/DataSet/Aggregated/DataSet_aggregated_demo_remove_none_1s.csv")
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

    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )

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

    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("===== Classification Report =====")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("===== Confusion Matrix =====")
    print(confusion_matrix(y_test, y_pred))

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    feature_names = []
    for i in range(FRAME_SIZE):
        for feat in FEATURES:
            feature_names.append(f"{feat}_{i}")
    feature_names = np.array(feature_names)

    print("\n上位の重要特徴量:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
