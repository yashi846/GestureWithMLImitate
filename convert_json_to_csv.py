"""
指定した複数の生データ(json)をそれぞれデータセット(csv)に変換するスクリプト
"""

import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

RAWDATA_ROOT = Path("RawData")
OUTPUT_ROOT = Path("DataSet")

# labelはファイル名のプレフィックスを除いた部分
GESTURE_CONFIGS = [
    # {"dir_name": "R2L", "label": "right2left"},
    # {"dir_name": "L2R", "label": "left2right"},
    # {"dir_name": "U2D", "label": "up2down"},
    # {"dir_name": "D2U", "label": "down2up"},
    {"dir_name": "None", "label": "none"},
]
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
    """ジェスチャラベルに対応する整数インデックスを返す"""
    for idx, config in enumerate(GESTURE_CONFIGS):
        if config["dir_name"] == gesture_label:
            return idx
    return -1


def convert_to_csv(gesture_config: Dict, output_dir: Path):
    """指定ジェスチャディレクトリのJSONをCSVに変換"""
    dir_name = gesture_config["dir_name"]
    label = gesture_config["label"]

    input_dir = RAWDATA_ROOT / dir_name
    if not input_dir.exists():
        print(f"[SKIP] ディレクトリが存在しません: {input_dir}")
        return

    file_groups = group_files_by_number(input_dir)

    if not file_groups:
        print(f"[SKIP] {dir_name}: 対象ファイルが見つかりません")
        return

    for file_number, json_files in file_groups.items():
        frames = load_json_data(json_files)

        if not frames:
            print(f"  [SKIP] {dir_name}/{file_number}: データが見つかりません")
            continue

        output_csv = output_dir / dir_name / f"{file_number}_{label}.csv"
        label_index = get_label_index(dir_name)

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            header = FEATURE_COLUMNS + [LABEL_COLUMN]
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for features in frames:
                if len(features) >= len(FEATURE_COLUMNS):
                    row = features[: len(FEATURE_COLUMNS)] + [label_index]
                    writer.writerow(row)


def main():
    for gesture_config in GESTURE_CONFIGS:
        convert_to_csv(gesture_config, OUTPUT_ROOT)
        print(f"出力先: {OUTPUT_ROOT / Path(gesture_config['label'])}")


if __name__ == "__main__":
    main()
