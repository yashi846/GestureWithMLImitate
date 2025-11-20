import re
import argparse
from pathlib import Path
import shutil
from typing import List

""" 
ファイル(複数可)を移動させデータセットとしてリネームし保存させるコード
複数ファイルを移動させる場合は、2.1_~~~ , 2.2_~~~ とリネームされる
"""


# 定数: ディレクトリ名とラベル文字列（ハードコード回避）
DATASET_ROOT = "DataSet"
GESTURE_DIR_NAME = "U2D"
GESTURE_LABEL = "up2down"
DEFAULT_TARGET_DIR = f"{DATASET_ROOT}/{GESTURE_DIR_NAME}"

PATTERN = re.compile(rf"^(\d+)(?:\.(\d+))?_{GESTURE_LABEL}(\.[^.]+)?$")

def preprocess_collect(collect_from: Path, target_dir: Path, do_execute: bool, mode: str, recursive: bool, exts: List[str], verbose: bool):
    """Collect files from collect_from into target_dir.

    - recursive: walk all subdirectories if True, else only direct children
    - exts: list of extensions to match (normalized to leading dot, lowercase)
    - mode: 'move' or 'copy'
    - verbose: list each candidate before action
    """
    if not collect_from.exists():
        print(f"collect-from が存在しません: {collect_from}")
        return
    norm_exts = []
    for e in exts:
        if not e:
            continue
        e = e.lower()
        if not e.startswith('.'):
            e = '.' + e
        norm_exts.append(e)
    candidates = []
    if recursive:
        for p in collect_from.rglob('*'):
            if p.is_file() and p.suffix.lower() in norm_exts:
                candidates.append(p)
    else:
        # files directly under collect_from
        for p in collect_from.iterdir():
            if p.is_file() and p.suffix.lower() in norm_exts:
                candidates.append(p)
        # one-level subdirectories
        for p in collect_from.iterdir():
            if p.is_dir():
                for item in p.iterdir():
                    if item.is_file() and item.suffix.lower() in norm_exts:
                        candidates.append(item)
    if not candidates:
        print(f"前処理: 対象拡張子 {norm_exts} のファイルが見つかりません (recursive={recursive}).")
        return
    print(f"=== 前処理候補 === {len(candidates)} files (mode={mode}, recursive={recursive}, exts={norm_exts})")
    moved_count = 0
    skipped_count = 0
    for src in candidates:
        dest = target_dir / src.name
        if dest.exists():
            print(f"[skip 重複] {dest.name}")
            skipped_count += 1
            continue
        if verbose:
            print(f"[candidate] {src} -> {dest}")
        if do_execute:
            if mode == 'move':
                try:
                    shutil.move(str(src), str(dest))
                except Exception as e:
                    print(f"[error move] {src}: {e}")
                    continue
            else:
                try:
                    shutil.copy2(str(src), str(dest))
                except Exception as e:
                    print(f"[error copy] {src}: {e}")
                    continue
        print(f"[collect {mode}{'' if do_execute else ' (dry)'}] {src.name}")
        moved_count += 1
    print(f"=== 前処理結果 === files: {moved_count} collected, {skipped_count} skipped (重複) {'(実行済)' if do_execute else '(ドライラン) --execute で確定)'}")

def parse_existing_numbers(files):
    integers = set()
    for f in files:
        m = PATTERN.match(f.name)
        if m:
            integers.add(int(m.group(1)))
    return integers

def generate_new_names(unmatched_files, existing_names, start_base):
    """Assign names: first gets base.0, rest base.1, base.2 ... avoiding collisions.

    base integer starts at start_base; if base.0 or base (legacy) already exists, increment.
    """
    assigned = {}
    if not unmatched_files:
        return assigned

    # Determine usable base (avoid collisions with either base_{GESTURE_LABEL} or base.0_{GESTURE_LABEL})
    candidate_base = start_base
    while True:
        legacy = f"{candidate_base}_{GESTURE_LABEL}.json"
        decimal0 = f"{candidate_base}.0_{GESTURE_LABEL}.json"
        # Accept any suffix by pattern; here we assume .json, but also check raw names without suffix just in case.
        if legacy in existing_names or decimal0 in existing_names:
            candidate_base += 1
            continue
        break

    base_used = candidate_base

    # First file: base.0
    first = unmatched_files[0]
    first_name = f"{base_used}.0_{GESTURE_LABEL}{first.suffix}" if first.suffix else f"{base_used}.0_{GESTURE_LABEL}"
    assigned[first] = first_name

    # Subsequent files: base.i starting at 1
    decimal = 1
    for f in unmatched_files[1:]:
        while True:
            name = f"{base_used}.{decimal}_{GESTURE_LABEL}{f.suffix}" if f.suffix else f"{base_used}.{decimal}_{GESTURE_LABEL}"
            if name not in existing_names and name not in assigned.values():
                assigned[f] = name
                decimal += 1
                break
            decimal += 1

    return assigned

def collect_files(target_dir: Path):
    return [p for p in target_dir.iterdir() if p.is_file()]

def main():
    parser = argparse.ArgumentParser(description=f"Rename files in {GESTURE_DIR_NAME} to {{n}}_{GESTURE_LABEL} / {{n.i}}_{GESTURE_LABEL} pattern.")
    parser.add_argument("--dir", default=DEFAULT_TARGET_DIR, help=f"ターゲットディレクトリ (既定: {DEFAULT_TARGET_DIR})")
    parser.add_argument("--execute", action="store_true", help="これを付けると実際にリネームを行う。付けない場合はドライラン")
    parser.add_argument("--collect-from", help="前処理: 指定ディレクトリ内ファイルをターゲットへフラット集約")
    parser.add_argument("--collect-mode", choices=["move", "copy"], default="move", help="集約方式 move=移動 copy=コピー (既定: move)")
    parser.add_argument("--recursive", action="store_true", help="再帰的にサブディレクトリを走査")
    parser.add_argument("--ext", default=".json", help="対象拡張子(カンマ区切り). 例: .json,.bin")
    parser.add_argument("--verbose", action="store_true", help="収集候補を詳細表示")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return

    # Optional preprocess: flatten all .json files from subdirectories of --collect-from into target_dir
    if args.collect_from:
        preprocess_collect(
            collect_from=Path(args.collect_from),
            target_dir=target_dir,
            do_execute=args.execute,
            mode=args.collect_mode,
            recursive=args.recursive,
            exts=[e.strip().lower() for e in args.ext.split(',') if e.strip()],
            verbose=args.verbose,
        )

    all_files = collect_files(target_dir)
    existing_names = {f.name for f in all_files}
    integers = parse_existing_numbers(all_files)
    next_base = (max(integers) + 1) if integers else 0

    unmatched = [f for f in all_files if not PATTERN.match(f.name)]
    if not unmatched:
        print("リネーム対象 (未パターン) ファイルはありません。")
        return

    # Stable ordering for deterministic decimal assignment
    unmatched.sort(key=lambda p: p.name)

    plan = generate_new_names(unmatched, existing_names, next_base)

    print("=== リネーム計画 ===")
    for src, dst in plan.items():
        print(f"{src.name} -> {dst}")

    if not args.execute:
        print("(ドライラン) --execute を付けると実行します。")
        return

    # Execute rename operations
    for src, dst in plan.items():
        new_path = src.with_name(dst)
        src.rename(new_path)
    print("リネーム完了。")

if __name__ == "__main__":
    main()
