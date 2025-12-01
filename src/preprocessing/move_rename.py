"""
ファイル(複数可)を移動させデータセットとしてリネームし保存させるコード
複数ファイルを移動させる場合は、2.1_~~~ , 2.2_~~~ とリネームされる
--collect-fromで、移動するディレクトリ名を指定する
"""

import re
import argparse
from pathlib import Path
import shutil


DATASET_ROOT = Path("../../RawData")
GESTURE_DIR_NAME = "None"
GESTURE_LABEL = "none"
DEFAULT_TARGET_DIR = Path(f"{DATASET_ROOT}/{GESTURE_DIR_NAME}")

PATTERN = re.compile(rf"^(\d+)(?:\.(\d+))?_{GESTURE_LABEL}(\.[^.]+)?$")


def preprocess_collect(
    collect_from: Path,
    target_dir: Path,
):
    if not collect_from.exists():
        print(f"collect-from が存在しません: {collect_from}")
        return

    candidates = []
    for p in collect_from.rglob("*.json"):
        if p.is_file():
            candidates.append(p)

    if not candidates:
        print(".json ファイルが見つかりません。")
        return

    for src in candidates:
        dest = target_dir / src.name
        try:
            shutil.move(str(src), str(dest))
        except Exception as e:
            print(f"[error move] {src}: {e}")
            continue
        print(f"[collect move] {src.name}")


def parse_existing_numbers(files):
    integers = set()
    for f in files:
        m = PATTERN.match(f.name)
        if m:
            integers.add(int(m.group(1)))
    return integers


def generate_new_names(unmatched_files, existing_names, start_base):
    assigned = {}
    if not unmatched_files:
        return assigned

    candidate_base = start_base
    while True:
        decimal0 = f"{candidate_base}.0_{GESTURE_LABEL}.json"
        if decimal0 in existing_names:
            candidate_base += 1
            continue
        break

    base_used = candidate_base

    first = unmatched_files[0]
    first_name = (
        f"{base_used}.0_{GESTURE_LABEL}{first.suffix}"
        if first.suffix
        else f"{base_used}.0_{GESTURE_LABEL}"
    )
    assigned[first] = first_name

    decimal = 1
    for f in unmatched_files[1:]:
        while True:
            name = (
                f"{base_used}.{decimal}_{GESTURE_LABEL}{f.suffix}"
                if f.suffix
                else f"{base_used}.{decimal}_{GESTURE_LABEL}"
            )
            if name not in existing_names and name not in assigned.values():
                assigned[f] = name
                decimal += 1
                break
            decimal += 1

    return assigned


def collect_files(target_dir: Path):
    return [p for p in target_dir.iterdir() if p.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description=f"Rename files in {GESTURE_DIR_NAME} to {{n}}_{GESTURE_LABEL} / {{n.i}}_{GESTURE_LABEL} pattern."
    )
    parser.add_argument(
        "--collect-from",
        help="移動するディレクトリ名を指定",
    )
    args = parser.parse_args()

    target_dir = DEFAULT_TARGET_DIR
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return

    if args.collect_from:
        preprocess_collect(
            collect_from=Path(args.collect_from),
            target_dir=target_dir,
        )

    all_files = collect_files(target_dir)
    existing_names = {f.name for f in all_files}
    integers = parse_existing_numbers(all_files)
    next_base = (max(integers) + 1) if integers else 0

    unmatched = [f for f in all_files if not PATTERN.match(f.name)]
    if not unmatched:
        print("リネーム対象 (未パターン) ファイルはありません。")
        return

    unmatched.sort(key=lambda p: p.name)

    plan = generate_new_names(unmatched, existing_names, next_base)

    print("=== リネーム計画 ===")
    for src, dst in plan.items():
        print(f"{src.name} -> {dst}")

    for src, dst in plan.items():
        new_path = src.with_name(dst)
        src.rename(new_path)
    print("リネーム完了。")


if __name__ == "__main__":
    main()
