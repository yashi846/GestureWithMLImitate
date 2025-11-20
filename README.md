# GestureWithMLImitate

ジェスチャ認識の機械学習プロジェクト

## 概要

IWR6843 AOPのデモ Gesture with Machine Learningを元にして
ジェスチャ認識の機械学習を行うプロジェクトです。


## ディレクトリ構成

```
.
├── RawData/              # 生のJSONデータ
│   ├── R2L/             # Right to Left (右から左)
│   ├── L2R/             # Left to Right (左から右)
│   ├── U2D/             # Up to Down (上から下)
│   ├── D2U/             # Down to Up (下から上)
│   ├── CWT/             # Clockwise Turn (時計回り)
│   ├── CCWT/            # Counter Clockwise Turn (反時計回り)
│   ├── Push/            # Push (押す)
│   ├── Pull/            # Pull (引く)
│   └── Shine/           # Shine (光る)
├── DataSet/             # 変換後のCSVデータ
│   ├── R2L/
│   ├── L2R/
│   ├── U2D/
│   └── D2U/
└── ProcessedData/       # 処理済みデータ（出力先）
```

## データ形式

### CSV 出力形式

| カラム名      | 説明                     | features 配列のインデックス |
| ------------- | ------------------------ | --------------------------- |
| wtDoppler     | ドップラー               | features[0]                 |
| wtDopplerPos  | 正のドップラー           | features[1]                 |
| wtDopplerNeg  | 負のドップラー           | features[2]                 |
| wtRange       | レンジ                   | features[3]                 |
| numDetections | 検出数                   | features[4]                 |
| wtAzimuthMean | 方位角平均               | features[5]                 |
| wtElevMean    | 仰角平均                 | features[6]                 |
| azDoppCorr    | 方位角ドップラー相関     | features[7]                 |
| wtAzimuthStd  | 方位角標準偏差           | features[8]                 |
| wtdElevStd    | 仰角標準偏差             | features[9]                 |
| label         | ジェスチャラベル（整数） | -                           |

### ラベル対応表

| ジェスチャ     | ラベル値 |
| -------------- | -------- |
| R2L (右から左) | 0        |
| L2R (左から右) | 1        |
| U2D (上から下) | 2        |
| D2U (下から上) | 3        |

## スクリプト

### convert_json_to_csv.py

JSON ファイルを CSV 形式に変換します。

### rename_l2r.py

IWR6843 AOPの測定で得たデータのディレクトリ内のファイルを移動させ、規則的な命名規則にリネームします。
