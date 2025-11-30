"""
Prototypical Networksで学習を行うスクリプト
NNにはMLPを使用している
以下のパラメータを最適化する
- 中間層のユニット数
- 埋め込み次元数
- ドロップアウト率
- 重み減衰
- 学習率
- Shot数
- クエリ数
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import optuna


def check_gpu():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU検出成功: {gpu_name} (Total: {gpu_count}台)")
        device = torch.device("cuda")
    else:
        print("GPUが見つかりません。CPUで実行します。")
        device = torch.device("cpu")
    return device


DEVICE = check_gpu()


CSV_PATH = Path("DataSet/Aggregated/DataSet_aggregated_demo_remove_none_1s.csv")
FEATURES = [
    "wtDoppler",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
]

FRAME_SIZE = 15
INPUT_DIM = len(FEATURES) * FRAME_SIZE


N_WAY = 4
N_TRIALS = 100


class GestureDataset(Dataset):
    # Datasetの役割:
    # - ラベルを整数IDに変換（LabelEncoder）
    # - 特徴量を標準化（StandardScaler）
    # - エピソードサンプリング用に「クラスごとのインデックス」を保持
    def __init__(self, df, scaler=None, label_encoder=None):
        # 必要な特徴量をチェック
        required_cols = [f"{feat}_{i}" for i in range(FRAME_SIZE) for feat in FEATURES]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"特徴量が不足しています。必要な特徴量: {missing}\n")

        feature_cols = required_cols
        features = df[feature_cols].values.astype(np.float32)
        label = df["label"].values

        # ラベルを整数IDに変換
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(label)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(label)

        self.classes = self.label_encoder.classes_
        self.n_classes = len(self.classes)

        # 特徴量の標準化
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features)

        # Tensor化(行列の一般化)してDEVICEへ（CPU/GPUどちらでも動くように）
        self.features = torch.FloatTensor(self.features).to(DEVICE)

        # 特定のクラスのサンプルを素早く取り出すために、辞書にキャッシュしておく
        # 例: class_indices[2] -> ラベルID=2のサンプルの位置一覧
        self.class_indices = {
            i: np.where(self.encoded_labels == i)[0] for i in range(self.n_classes)
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.encoded_labels[idx]


class EpisodeSampler:
    # Few-shot学習の「エピソード」を作るクラス
    # 1エピソード = N_WAY個のクラスをランダムに選び、各クラスから
    #   - k_shot: サポート
    #   - n_query: クエリ
    # を取り出す。
    #
    # 例:
    #   n_way=3, k_shot=2, n_query=2 のとき
    #   クラスA,B,Cから A:2+2, B:2+2, C:2+2 サンプルを集め、
    #   サポートの平均=プロトタイプを作り、クエリは最も近いプロトタイプへ分類。
    def __init__(self, dataset, n_way, k_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def sample_episode(self):
        if self.dataset.n_classes < self.n_way:
            return None

        # クラスのインデックスを N_WAY 個選ぶ（例: [0, 3, 5, 7]）
        selected_classes = np.random.choice(
            self.dataset.n_classes, size=self.n_way, replace=False
        )

        sx, sy, qx, qy = [], [], [], []

        for local_lbl, global_idx in enumerate(selected_classes):
            indices = self.dataset.class_indices[global_idx]
            n_needed = self.k_shot + self.n_query

            if len(indices) == 0:
                continue

            # サンプル不足時は重複あり抽出
            replace = len(indices) < n_needed
            selected = np.random.choice(indices, size=n_needed, replace=replace)

            # サポートとクエリを分割
            sx.append(self.dataset.features[selected[: self.k_shot]])
            qx.append(self.dataset.features[selected[self.k_shot :]])

            # 学習時の損失計算はローカルなクラスIDで行う
            sy.append(torch.tensor([local_lbl] * self.k_shot, device=DEVICE))
            qy.append(torch.tensor([local_lbl] * self.n_query, device=DEVICE))

        if not sx:
            return None

        # 評価時にローカルからグローバルな整数IDへ戻せるようにする
        return (
            torch.cat(sx),
            torch.cat(sy),
            torch.cat(qx),
            torch.cat(qy),
            selected_classes,
        )


class ProtoEncoder(nn.Module):
    # シンプルなMLPで入力特徴から埋め込みベクトル（プロトタイプ空間）へ写像
    # BatchNorm/Dropoutで汎化性能の向上を狙う
    def __init__(self, input_dim, hidden1, hidden2, embed_dim, dropout):
        super().__init__()
        layers = []
        layers.extend(
            [
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        layers.extend(
            [
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        layers.append(nn.Linear(hidden2, embed_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, input_dim] -> [batch_size, embed_dim]
        return self.encoder(x)


def compute_prototypes(support, labels, n_way):
    # プロトタイプ（各クラスの埋め込みの平均）を計算
    # ミニ例:
    #   support = [[...], [...], ...], labels = [0,0,1,1,2,2], n_way=3
    #   -> クラス0の平均, クラス1の平均, クラス2の平均 を返す
    protos = []
    for i in range(n_way):
        mask = labels == i  # クラスiの位置だけTrue
        if mask.sum() > 0:
            protos.append(support[mask].mean(dim=0))
        else:
            # サンプルが不足する場合のフォールバック（ゼロベクトル）
            protos.append(torch.zeros(support.size(1), device=DEVICE))
    return torch.stack(protos)


df = pd.read_csv(CSV_PATH)
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

train_ds = GestureDataset(train_df)
# テスト側は訓練のScaler/Encoderを共有
# データリーク（train の統計を test 側で不適切に再推定して精度を過大評価してしまう現象）防止の基本形
test_ds = GestureDataset(
    test_df, scaler=train_ds.scaler, label_encoder=train_ds.label_encoder
)


def objective(trial):
    # Optunaで探索するハイパーパラメータを定義
    # - ネットワーク構成（hidden1, hidden2, embed_dim）
    # - 正則化（dropout, weight_decay）
    # - 学習率（lr）
    # - Few-shot設定（k_shot, n_query）
    hidden1 = trial.suggest_categorical("hidden1", [128, 256, 512])
    hidden2 = trial.suggest_categorical("hidden2", [64, 128, 256])
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    k_shot = trial.suggest_categorical("k_shot", [5, 10])
    n_query = trial.suggest_categorical("n_query", [5, 10, 15])

    epochs = 60

    train_sampler = EpisodeSampler(train_ds, N_WAY, k_shot, n_query)
    test_sampler = EpisodeSampler(test_ds, N_WAY, k_shot, n_query)

    model = ProtoEncoder(INPUT_DIM, hidden1, hidden2, embed_dim, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        for _ in range(30):  # 1エポックあたりのエピソード数
            batch = train_sampler.sample_episode()
            if batch is None:
                continue
            sx, sy, qx, qy, _ = batch
            optimizer.zero_grad()
            s_emb, q_emb = model(sx), model(qx)
            protos = compute_prototypes(s_emb, sy, N_WAY)
            dists = torch.cdist(q_emb, protos)
            loss = F.cross_entropy(-dists, qy)
            loss.backward()
            optimizer.step()

        # --- 評価（プルーニング用） ---
        # テストエピソードでの平均精度を報告。低調なら早期打ち切り。
        model.eval()
        total_acc = 0.0
        eval_episodes = 15
        with torch.no_grad():
            for _ in range(eval_episodes):
                batch = test_sampler.sample_episode()
                if batch is None:
                    continue
                sx, sy, qx, qy, _ = batch

                s_emb, q_emb = model(sx), model(qx)
                protos = compute_prototypes(s_emb, sy, N_WAY)
                dists = torch.cdist(q_emb, protos)
                preds = torch.argmin(dists, dim=1)

                acc = (preds == qy).float().mean().item()
                total_acc += acc

        avg_acc = total_acc / eval_episodes
        trial.report(avg_acc, epoch)
        if trial.should_prune():
            # 途中成績が悪ければ計算を途中打ち切り
            raise optuna.exceptions.TrialPruned()

    return avg_acc


if __name__ == "__main__":
    # MedianPrunerにより中間成績が中央値より悪い試行は早期終了
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 40)
    print(f"ベストスコア (Val Acc): {study.best_value:.4f}")
    print("ベストパラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 40 + "\n")

    print("ベストパラメータでモデルを再学習・評価")

    bp = study.best_params

    final_train_sampler = EpisodeSampler(train_ds, N_WAY, bp["k_shot"], bp["n_query"])
    final_test_sampler = EpisodeSampler(test_ds, N_WAY, bp["k_shot"], bp["n_query"])

    best_model = ProtoEncoder(
        INPUT_DIM, bp["hidden1"], bp["hidden2"], bp["embed_dim"], bp["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        best_model.parameters(), lr=bp["lr"], weight_decay=bp["weight_decay"]
    )

    final_epochs = 60
    best_model.train()
    for epoch in range(final_epochs):
        for _ in range(50):
            batch = final_train_sampler.sample_episode()
            if batch is None:
                continue
            sx, sy, qx, qy, _ = batch

            optimizer.zero_grad()
            s_emb, q_emb = best_model(sx), best_model(qx)
            protos = compute_prototypes(s_emb, sy, N_WAY)
            dists = torch.cdist(q_emb, protos)
            loss = F.cross_entropy(-dists, qy)
            loss.backward()
            optimizer.step()

    best_model.eval()
    all_true_global = []
    all_pred_global = []

    test_episodes = 100

    with torch.no_grad():
        for _ in range(test_episodes):
            batch = final_test_sampler.sample_episode()
            if batch is None:
                continue
            sx, sy, qx, qy, selected_classes = batch

            s_emb, q_emb = best_model(sx), best_model(qx)
            protos = compute_prototypes(s_emb, sy, N_WAY)
            dists = torch.cdist(q_emb, protos)

            preds_local = torch.argmin(dists, dim=1).cpu().numpy()
            true_local = qy.cpu().numpy()

            preds_global = selected_classes[preds_local]
            true_global = selected_classes[true_local]

            all_true_global.extend(true_global)
            all_pred_global.extend(preds_global)

    class_names = list(test_ds.label_encoder.classes_)

    print("===== Classification Report =====")
    print(
        classification_report(
            all_true_global, all_pred_global, target_names=class_names
        )
    )

    print("===== Confusion Matrix =====")
    cm = confusion_matrix(all_true_global, all_pred_global)
    print(cm)

    # df_results = study.trials_dataframe()
    # df_results.to_csv("optuna_results_optimized.csv")
    # print("探索履歴を 'optuna_results_optimized.csv' に保存しました。")
