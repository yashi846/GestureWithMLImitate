import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# 定数: 対象ディレクトリとラベル設定（リファクタリング用）
RAWDATA_ROOT = "RawData"
OUTPUT_ROOT = "DataSet"
GESTURE_CONFIGS = [
    {"dir_name": "R2L", "label": "right2left"},
    {"dir_name": "L2R", "label": "left2right"},
    {"dir_name": "U2D", "label": "up2down"},
    {"dir_name": "D2U", "label": "down2up"},
]

# CSVカラム名（featuresの各要素に対応）
FEATURE_COLUMNS = [
    "wtDoppler",
    "wtDopplerPos",
    "wtDopplerNeg",
    "wtRange",
    "numDetections",
    "wtAzimuthMean",
    "wtElevMean",
    "azDoppCorr",
    "wtAzimuthStd",
    "wtdElevStd",
]

# ラベルカラム名
LABEL_COLUMN = "label"


def extract_file_number(filename: str) -> str:
    """ファイル名から通し番号を抽出 (例: '5.0_xxx.json' -> '5', '5.1_xxx.json' -> '5')"""
    match = re.match(r"^(\d+)(?:\.\d+)?_", filename)
    if match:
        return match.group(1)
    return None


def group_files_by_number(directory: Path) -> Dict[str, List[Path]]:
    """通し番号でファイルをグループ化"""
    groups = defaultdict(list)
    for file_path in sorted(directory.glob("*.json")):
        file_number = extract_file_number(file_path.name)
        if file_number:
            groups[file_number].append(file_path)
    return groups


def load_json_data(json_files: List[Path]) -> List[Dict]:
    """複数のJSONファイルから全データフレームを抽出"""
    all_frames = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"]:
                    if "frameData" in item and "features" in item["frameData"]:
                        all_frames.append(item["frameData"]["features"])
    return all_frames


def get_label_index(gesture_label: str) -> int:
    """ジェスチャラベルに対応する整数インデックスを返す (0, 1, 2, 3)"""
    for idx, config in enumerate(GESTURE_CONFIGS):
        if config["dir_name"] == gesture_label:
            return idx
    return -1


def convert_to_csv(gesture_config: Dict, output_dir: Path):
    """指定ジェスチャディレクトリのJSONをCSVに変換"""
    dir_name = gesture_config["dir_name"]
    label = gesture_config["label"]
    
    input_dir = Path(RAWDATA_ROOT) / dir_name
    if not input_dir.exists():
        print(f"[SKIP] ディレクトリが存在しません: {input_dir}")
        return
    
    # ファイルを通し番号でグループ化
    file_groups = group_files_by_number(input_dir)
    
    if not file_groups:
        print(f"[SKIP] {dir_name}: 対象ファイルが見つかりません")
        return
    
    print(f"[{dir_name}] {len(file_groups)} グループのファイルを処理中...")
    
    # 出力ディレクトリを作成
    output_subdir = output_dir / dir_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 各グループをCSVに変換
    for file_number, json_files in file_groups.items():
        # JSONデータを読み込み
        frames = load_json_data(json_files)
        
        if not frames:
            print(f"  [SKIP] {dir_name}/{file_number}: データが見つかりません")
            continue
        
        # CSV出力
        output_csv = output_subdir / f"{file_number}_{label}.csv"
        label_index = get_label_index(dir_name)
        
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            # ヘッダー作成
            header = FEATURE_COLUMNS + [LABEL_COLUMN]
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            # データ行を書き込み
            for features in frames:
                if len(features) >= len(FEATURE_COLUMNS):
                    row = features[:len(FEATURE_COLUMNS)] + [label_index]
                    writer.writerow(row)
        
        print(f"  [OK] {output_csv.name} ({len(frames)} rows)")


def main():
    """メイン処理"""
    # 出力ディレクトリ
    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(exist_ok=True)
    
    print("=== JSON to CSV 変換開始 ===\n")
    
    # 各ジェスチャディレクトリを処理
    for gesture_config in GESTURE_CONFIGS:
        convert_to_csv(gesture_config, output_dir)
    
    print("\n=== 変換完了 ===")
    print(f"出力先: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
