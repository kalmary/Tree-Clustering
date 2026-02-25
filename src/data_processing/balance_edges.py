import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def balance_graph_files_by_edge_ratio(
    edges_dir: Path,
    split: str = 'train',
    dry_run: bool = False,
    backup: bool = False,
    verbose: bool = True,
    min_minority_ratio: float = 0.20,
    target_ratio: float = 1.0,
) -> dict:
    """
    Balance graph files by:
        1. Removing files where minority class < min_minority_ratio (hopeless files).
        2. Among remaining files, removing those with the most majority-class edges
           first until the global target_ratio is reached.

    Args:
        edges_dir:          Directory containing split subdirectories.
        split:              Which split to process.
        dry_run:            Only report, don't modify.
        backup:             Rename to .pt.bak instead of deleting.
        verbose:            Show progress and stats.
        min_minority_ratio: Files with minority ratio below this are always removed.
        target_ratio:       Target majority/minority ratio globally.
                            1.0 = equal, 2.0 = allow 2x more majority than minority.
    """
    edges_dir = Path(edges_dir) / split

    if not edges_dir.exists():
        raise ValueError(f"Directory does not exist: {edges_dir}")

    pt_files = sorted(edges_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {edges_dir}")
        return {}

    if verbose:
        print(f"\n{'='*80}")
        print(f"BALANCE GRAPH FILES - {split.upper()} (BY EDGE RATIO)")
        print(f"{'='*80}\n")
        print(f"Found {len(pt_files)} files")
        print(f"Step 1: removing files with <{min_minority_ratio*100:.0f}% minority class")
        print(f"Step 2: removing worst majority-dominated files until {target_ratio:.1f}:1 ratio")
        print(f"\nScanning files...")

    stats = {
        'files_processed':     0,
        'files_kept':          0,
        'files_removed':       0,
        'original_counts':     defaultdict(int),
        'final_counts':        defaultdict(int),
        'imbalance_histogram': defaultdict(int),
    }

    hopeless   = []   # below min_minority_ratio — always removed
    candidates = []   # above threshold — may be removed in step 2

    pbar = tqdm(pt_files, desc="Scanning files", disable=not verbose)
    for file_path in pbar:
        try:
            graph   = torch.load(file_path, map_location='cpu', weights_only=True)
            labels  = graph['y']
            class_0 = int((labels == 0).sum())
            class_1 = int((labels == 1).sum())
            total   = class_0 + class_1
            del graph

            stats['original_counts'][0] += class_0
            stats['original_counts'][1] += class_1
            stats['files_processed']    += 1

            minority_count = min(class_0, class_1)
            majority_count = max(class_0, class_1)
            minority_ratio = minority_count / total if total > 0 else 0.0

            bin_idx = min(int(minority_ratio * 10) * 10, 90)
            stats['imbalance_histogram'][bin_idx] += 1

            file_info = {
                'path':           file_path,
                'class_0':        class_0,
                'class_1':        class_1,
                'total':          total,
                'minority_ratio': minority_ratio,
                'majority_count': majority_count,
            }

            if minority_ratio < min_minority_ratio:
                hopeless.append(file_info)
            else:
                candidates.append(file_info)

        except Exception as e:
            if verbose:
                pbar.write(f"Error processing {file_path.name}: {e}")

    # --- step 2: among candidates, remove worst majority offenders until target ---
    # sort candidates by majority_count descending — worst offenders first
    candidates.sort(key=lambda f: f['majority_count'], reverse=True)

    # compute current totals excluding hopeless files
    cand_c0 = sum(f['class_0'] for f in candidates)
    cand_c1 = sum(f['class_1'] for f in candidates)
    global_majority  = 1 if cand_c1 >= cand_c0 else 0
    global_minority  = 1 - global_majority
    running_majority = cand_c1 if global_majority == 1 else cand_c0
    running_minority = cand_c0 if global_majority == 1 else cand_c1
    majority_budget  = int(running_minority * target_ratio)

    keep_files   = []
    remove_step2 = []

    for f in candidates:
        f_majority = f['class_1'] if global_majority == 1 else f['class_0']
        if running_majority > majority_budget:
            remove_step2.append(f)
            running_majority -= f_majority
        else:
            keep_files.append(f)
            stats['final_counts'][0] += f['class_0']
            stats['final_counts'][1] += f['class_1']

    remove_files = hopeless + remove_step2
    stats['files_kept']    = len(keep_files)
    stats['files_removed'] = len(remove_files)

    if verbose:
        print(f"\nResults:")
        print(f"  Removed (minority ratio < {min_minority_ratio*100:.0f}%): {len(hopeless)}")
        print(f"  Removed (majority excess)                        : {len(remove_step2)}")
        print(f"  Files kept                                       : {len(keep_files)}  ({len(keep_files)/len(pt_files)*100:.1f}%)")

        print(f"\nMinority class ratio distribution (before removal):")
        for bin_val in sorted(stats['imbalance_histogram'].keys()):
            count = stats['imbalance_histogram'][bin_val]
            pct   = count / stats['files_processed'] * 100
            bar   = '█' * int(pct / 2)
            label = f"{bin_val:3d}-{min(bin_val + 10, 100):3d}%"
            print(f"  {label}: {count:5d} files ({pct:5.1f}%) {bar}")

    # --- remove / backup ---
    if not dry_run and remove_files:
        if verbose:
            print(f"\nRemoving {len(remove_files)} files...")
        for file_info in tqdm(remove_files, desc="Removing files", disable=not verbose):
            file_path = file_info['path']
            if backup:
                file_path.rename(file_path.with_suffix('.pt.bak'))
            else:
                file_path.unlink()

    # --- summary ---
    if verbose:
        total_orig  = stats['original_counts'][0] + stats['original_counts'][1]
        total_final = stats['final_counts'][0]     + stats['final_counts'][1]

        print(f"\n{'='*80}")
        print(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        print(f"{'='*80}\n")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files kept     : {stats['files_kept']}  ({stats['files_kept']/stats['files_processed']*100:.1f}%)")
        print(f"Files removed  : {stats['files_removed']}  ({stats['files_removed']/stats['files_processed']*100:.1f}%)")

        if total_orig > 0:
            print(f"\nOriginal distribution:")
            print(f"  Class 0: {stats['original_counts'][0]:>12,}  ({stats['original_counts'][0]/total_orig*100:.1f}%)")
            print(f"  Class 1: {stats['original_counts'][1]:>12,}  ({stats['original_counts'][1]/total_orig*100:.1f}%)")
            if stats['original_counts'][0] > 0:
                print(f"  Ratio  : {stats['original_counts'][1]/stats['original_counts'][0]:.2f}:1  (class1/class0)")

        if total_final > 0:
            print(f"\nFinal distribution:")
            print(f"  Class 0: {stats['final_counts'][0]:>12,}  ({stats['final_counts'][0]/total_final*100:.1f}%)")
            print(f"  Class 1: {stats['final_counts'][1]:>12,}  ({stats['final_counts'][1]/total_final*100:.1f}%)")
            if stats['final_counts'][0] > 0:
                orig_ratio  = stats['original_counts'][1] / stats['original_counts'][0]
                final_ratio = stats['final_counts'][1]    / stats['final_counts'][0]
                improvement = (orig_ratio - final_ratio) / orig_ratio * 100
                print(f"  Ratio  : {final_ratio:.2f}:1  (class1/class0)")
                print(f"\nImprovement: ratio reduced by {improvement:.1f}%")

        if dry_run:
            print("\nDry run - no files modified")

    return stats


if __name__ == "__main__":
    edges_dir = Path("data/edges")
  
    for split in ['train', 'val', 'test']:
        if (edges_dir / split).exists():
            balance_graph_files_by_edge_ratio(
                edges_dir,
                split=split,
                dry_run=False,
                backup=False,
                min_minority_ratio=0.1,
                target_ratio=3.0,
            )
