# GestureWithMLImitate

ジェスチャ認識の機械学習プロジェクト

## 概要

IWR6843 AOP のデモ Gesture with Machine Learning を元にしてジェスチャ認識の機械学習を行うプロジェクト

## ディレクトリ構成

```
.
├── .dvc/                # DVC設定ファイル
├── .vscode/             # VSCode設定
├── data/                # データディレクトリ（DVCで管理）
│   ├── RawData/        # 生のJSONデータ
│   ├── DataSet/        # 生データを変換したCSVデータ
│   │   ├── Aggregated/ # データセットを1つのファイルにまとめたデータ
│   │   └── ExtractGesture/ # ジェスチャーをしている最中を抽出したデータ
├── src/                # ソースコード
│   ├── models/ # 機械学習のモデルのスクリプト
│   ├── preprocessing/ # データの前処理のスクリプト
├── .gitignore
├── .python-version
├── dvc.yaml            # DVCパイプライン定義
├── pyproject.toml      # Poetry設定ファイル
├── poetry.lock
├── ruff.toml           # Ruff設定ファイル
└── README.md
```

## 略語の説明

### ジェスチャの種類

| ジェスチャ名 | 説明                                |
| ------------ | ----------------------------------- |
| None         | No Gesture (ジェスチャなし)         |
| R2L          | Right to Left (右から左)            |
| L2R          | Left to Right (左から右)            |
| U2D          | Up to Down (上から下)               |
| D2U          | Down to Up (下から上)               |
| CWT          | Clockwise Turn (時計回り)           |
| CCWT         | Counter Clockwise Turn (反時計回り) |
| Push         | Push (押す)                        |
| Pull         | Pull (引く)                        |
| Shine        | Shine (グーパー)                    |

### 特徴量の種類

| 特徴量      | 説明                 |
| ------------- | -------------------- |
| wtDoppler     | ドップラー           |
| wtDopplerPos  | 正のドップラー       |
| wtDopplerNeg  | 負のドップラー       |
| wtRange       | レンジ               |
| numDetections | 検出数               |
| wtAzimuthMean | 方位角平均           |
| wtElevMean    | 仰角平均             |
| azDoppCorr    | 方位角ドップラー相関 |
| wtAzimuthStd  | 方位角標準偏差       |
| wtdElevStd    | 仰角標準偏差         |
| label         | ジェスチャラベル     |

## スクリプト

### move_rename.py

IWR6843 AOP の測定で得たデータを移動させ、規則的な命名規則にリネームします。

**オプション:**

- `--collect-from`: 移動元ディレクトリを指定

**使用方法:**

```bash
python move_rename.py --collect-from <ソースディレクトリ>
```

**命名規則:**

- 単一ファイル: `{n}.0_{gesture}.json`
- 複数ファイル: `{n}.0_{gesture}.json`, `{n}.1_{gesture}.json`, ...

### convert_json_to_csv.py

JSON ファイルを CSV 形式に変換します。
RawData 内の JSON ファイルを DataSet ディレクトリに CSV として出力します。

### analyze_wt_doppler.py

wtDoppler データの統計分析を行い、ヒストグラムを生成します。

**機能:**

- DataSet/None 内の全 CSV ファイルから wtDoppler データを収集
- 平均、不偏標準偏差を計算
- ヒストグラムを生成

**出力:**

- `DataSet/Aggregated/wtDoppler_hist.png`: ヒストグラム画像
- 統計情報（平均、標準偏差、95%/70%範囲）をコンソール出力

### extract_gesture_moments.py

CSV データからジェスチャが実際に行われている瞬間を抽出します。

**機能:**

- numDetections が閾値以上の連続区間を検出
- 抽出したデータを DataSet/ExtractGesture に出力

### aggregate_csv.py

DataSet 内の各ジェスチャディレクトリの CSV ファイルを 1 つのファイルに統合します。

**機能:**

- 各ジェスチャディレクトリ内の全 CSV を結合
- 統合ファイルを DataSet/Aggregated に出力

### train_mlp.py

集約されたデータセットを多層パーセプトロン(MLP)で学習します。

**機能:**

1. 集約済み CSV を読み込み
2. ラベルをダミー変数に変換
3. 学習データとテストデータに分割
4. 特徴量を標準化
5. MLP で学習
6. テストデータで精度を評価
7. Classification Report と Confusion Matrix を出力

### train_prototypical_MLP.py

Prototypical Networks を用いて MLP で埋め込みを学習し、評価・最適化を行います。

**最適化対象:**

- 中間層ユニット数（hidden1, hidden2）
- 埋め込み次元（embed_dim）
- ドロップアウト率（dropout）
- 重み減衰（weight_decay）
- 学習率（lr）
- Shot 数（k_shot）
- クエリ数（n_query）

**機能:**

1. 集約済み CSV を読み込み
2. 特徴量標準化・ラベルエンコード
3. エピソードサンプリング（N-WAY クラスから support/query 抽出）
4. MLP エンコーダで埋め込み生成
5. プロトタイプ（support 平均）と距離で分類
6. 探索とプルーニング
7. ベスト設定で再学習し Classification Report と Confusion Matrix を出力

### train_svm.py

SVM を用いて学習し、評価・最適化を行います。

**最適化対象:**

- C (正則化パラメータ)
- γ

**機能:**

1. 集約済み CSV を読み込み
2. 特徴量標準化・ラベルエンコード
3. SVMとCross Validationを行う
6. 探索とプルーニング
7. ベスト設定で再学習し Classification Report と Confusion Matrix を出力


## 開発環境

### Formatter/Linter

Ruff を使用

**フォーマット:**

```bash
ruff format
```

**Lint:**

```bash
ruff check
```

### パッケージマネージャー

poetry を使用

### データ管理

大規模データのバージョン管理に DVC を使用

### DVC の基本コマンド

```bash
# データのトラッキング
dvc add data/

# リモートストレージへのプッシュ
dvc push

# リモートストレージからの取得
dvc pull

# トラッキング中のデータの状態確認
dvc status
```

**注意:** `data/` ディレクトリは DVC で管理されているため、Git には `.dvc` ファイルのみがコミットされます。実際のデータファイルは `.gitignore` で除外されています.
