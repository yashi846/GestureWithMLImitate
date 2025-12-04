"""
Prototypical Networks + LSTMで学習を行うスクリプト
NNにはLSTMを使用している
以下のパラメータを最適化する
- LSTMの隠れ層の次元数
- LSTMの層数
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

FRAME_SIZE = 20
CSV_PATH = Path(f"data/DataSet/Aggregated/DA_{FRAME_SIZE}frames_remove_none_1s.csv")
FEATURES = [
    "wtDoppler",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
]
INPUT_DIM = len(FEATURES) * FRAME_SIZE
FEATURE_DIM = len(FEATURES)

N_WAY = 4


class GestureDataset(Dataset):
    def __init__(self, df, scaler=None, label_encoder=None):
        feature_cols = [f"{feat}_{i}" for i in range(FRAME_SIZE) for feat in FEATURES]

        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values

        # ラベルエンコーディング (共通のEncoderを使用)
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(y)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(y)

        self.classes = self.label_encoder.classes_
        self.n_classes = len(self.classes)

        # 標準化 (学習データのScalerを使い回す)
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(X)

        self.features = torch.FloatTensor(self.features).to(DEVICE)

        # クラスごとのインデックス管理
        self.class_indices = {}
        for i in range(self.n_classes):
            self.class_indices[i] = np.where(self.encoded_labels == i)[0]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.encoded_labels[idx]


class EpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

    def sample_episode(self):
        if self.dataset.n_classes < self.n_way:
            return None

        # N-wayのクラスをランダム選択
        selected_classes = np.random.choice(
            self.dataset.n_classes, size=self.n_way, replace=False
        )

        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for local_label, global_class_idx in enumerate(selected_classes):
            class_sample_indices = self.dataset.class_indices[global_class_idx]

            # サンプル数が足りない場合の処理
            n_needed = self.k_shot + self.n_query
            if len(class_sample_indices) == 0:
                continue

            replace_flag = len(class_sample_indices) < n_needed

            selected_indices = np.random.choice(
                class_sample_indices, size=n_needed, replace=replace_flag
            )

            s_idx = selected_indices[: self.k_shot]
            q_idx = selected_indices[self.k_shot :]

            support_x.append(self.dataset.features[s_idx])
            query_x.append(self.dataset.features[q_idx])

            # ローカルラベル (0 ~ N_WAY-1) を使用
            support_y.append(torch.tensor([local_label] * self.k_shot, device=DEVICE))
            query_y.append(torch.tensor([local_label] * self.n_query, device=DEVICE))

        if not support_x:
            return None

        # selected_classes は評価時のGlobal Label復元用に返す
        return (
            torch.cat(support_x),
            torch.cat(support_y),
            torch.cat(query_x),
            torch.cat(query_y),
            selected_classes,
        )


class ProtoLSTM(nn.Module):
    def __init__(self, feature_dim, lstm_hidden, lstm_layers, embed_dim, dropout):
        super(ProtoLSTM, self).__init__()

        # LSTM層
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )

    def forward(self, x):
        # x: (batch_size, 90) -> (batch_size, 15, 6) に変形
        # 時系列情報として扱うため
        x = x.view(-1, FRAME_SIZE, FEATURE_DIM)

        # LSTMに通す
        # out: (batch, seq, hidden), (hn, cn)
        out, (hn, cn) = self.lstm(x)

        # 最終時刻の隠れ状態を使用 (Many-to-One)
        # hn[-1] shape: (batch, hidden)
        last_hidden = hn[-1]

        return self.fc(last_hidden)


def compute_prototypes(support_embeddings, support_labels, n_way):
    prototypes = []
    for i in range(n_way):
        mask = support_labels == i
        if mask.sum() > 0:
            proto = support_embeddings[mask].mean(dim=0)
        else:
            proto = torch.zeros(support_embeddings.size(1)).to(DEVICE)
        prototypes.append(proto)
    return torch.stack(prototypes)


def objective(trial):
    # パラメータ探索範囲
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 128, 256])
    lstm_layers = trial.suggest_int("lstm_layers", 1, 2)
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    k_shot = trial.suggest_categorical("k_shot", [5, 10, 20])
    n_query = trial.suggest_categorical("n_query", [5, 10, 15])

    epochs = 50
    episodes_per_epoch = 50

    # データ準備
    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    train_ds = GestureDataset(train_df)
    test_ds = GestureDataset(
        test_df, scaler=train_ds.scaler, label_encoder=train_ds.label_encoder
    )

    train_sampler = EpisodeSampler(train_ds, N_WAY, k_shot, n_query)
    test_sampler = EpisodeSampler(test_ds, N_WAY, k_shot, n_query)

    # モデル構築 (ProtoLSTM)
    model = ProtoLSTM(FEATURE_DIM, lstm_hidden, lstm_layers, embed_dim, dropout).to(
        DEVICE
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学習ループ
    for epoch in range(epochs):
        model.train()
        for _ in range(episodes_per_epoch):
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

        # 枝刈りと評価
        model.eval()
        total_acc = 0.0
        eval_episodes = 20
        with torch.no_grad():
            for _ in range(eval_episodes):
                batch = test_sampler.sample_episode()
                if batch is None:
                    continue
                sx, sy, qx, qy, _ = batch
                s_emb, q_emb = model(sx), model(qx)
                protos = compute_prototypes(s_emb, sy, N_WAY)
                dists = torch.cdist(q_emb, protos)
                acc = (torch.argmin(dists, dim=1) == qy).float().mean().item()
                total_acc += acc

        avg_acc = total_acc / eval_episodes
        trial.report(avg_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_acc


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=100)

    print("\n========================================")
    print(f"ベストスコア (Val Acc): {study.best_value:.4f}")
    print("ベストパラメータ:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("========================================")

    # ベストパラメータでモデルを再学習・評価
    best_params = study.best_params

    df = pd.read_csv(CSV_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    train_ds = GestureDataset(train_df)
    test_ds = GestureDataset(
        test_df, scaler=train_ds.scaler, label_encoder=train_ds.label_encoder
    )

    final_train_sampler = EpisodeSampler(
        train_ds, N_WAY, best_params["k_shot"], best_params["n_query"]
    )
    final_test_sampler = EpisodeSampler(
        test_ds, N_WAY, best_params["k_shot"], best_params["n_query"]
    )

    best_model = ProtoLSTM(
        FEATURE_DIM,
        best_params["lstm_hidden"],
        best_params["lstm_layers"],
        best_params["embed_dim"],
        best_params["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        best_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    epochs = 100
    episodes_per_epoch = 50

    for epoch in range(epochs):
        best_model.train()
        for _ in range(episodes_per_epoch):
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
            all_true_global, all_pred_global, target_names=class_names, digits=4
        )
    )

    print("===== Confusion Matrix =====")
    print(confusion_matrix(all_true_global, all_pred_global))
