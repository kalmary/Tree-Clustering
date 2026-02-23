import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def balance_dataset_globally(
    edges_dir: Path,
    split: str = 'train',
    target_ratio: float = 1.0,   # class0 : class1 — 1.0 = equal, 0.5 = half as many class0
    dry_run: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Balance the dataset globally by removing excess class-1-dominated files.

    Strategy:
        1. Scan all files, record class 0 and class 1 edge counts.
        2. Sort files by minority (class 0) ratio descending — keep the most
           balanced files first.
        3. Greedily accumulate files until class 1 count would exceed
           target_ratio * total_class_0. Files that would push class 1 over
           the budget are removed.

    This preserves all files that contribute class 0 edges and preferentially
    removes files that are pure or near-pure class 1, minimising information
    loss while improving global balance.

    Args:
        edges_dir:    Root directory containing split subdirectories.
        split:        Which split to process ('train', 'val', 'test').
        target_ratio: Desired class1 / class0 ratio after balancing.
                      1.0 = equal counts, 2.0 = 2× more class1 than class0.
        dry_run:      If True, report only — do not delete files.
        verbose:      Show progress and stats.

    Returns:
        dict with stats.
    """
    split_dir = Path(edges_dir) / split
    if not split_dir.exists():
        raise ValueError(f"Directory does not exist: {split_dir}")

    pt_files = sorted(split_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {split_dir}")
        return {}

    # --- scan ---
    file_stats = []
    pbar = tqdm(pt_files, desc="Scanning", disable=not verbose)
    for path in pbar:
        try:
            graph  = torch.load(path, map_location='cpu', weights_only=True)
            labels = graph['y']
            c0     = int((labels == 0).sum())
            c1     = int((labels == 1).sum())
            total  = c0 + c1
            file_stats.append({
                'path':           path,
                'c0':             c0,
                'c1':             c1,
                'total':          total,
                'minority_ratio': c0 / total if total > 0 else 0.0,
            })
            del graph
        except Exception as e:
            if verbose:
                pbar.write(f"Error: {path.name}: {e}")

    total_c0 = sum(f['c0'] for f in file_stats)
    total_c1 = sum(f['c1'] for f in file_stats)

    if verbose:
        print(f"\nOriginal: {len(file_stats)} files")
        print(f"  Class 0: {total_c0:>12,}  ({total_c0/(total_c0+total_c1)*100:.1f}%)")
        print(f"  Class 1: {total_c1:>12,}  ({total_c1/(total_c0+total_c1)*100:.1f}%)")
        print(f"  Ratio  : {total_c1/total_c0:.2f}:1  (target {target_ratio:.2f}:1)")

    # --- greedy selection ---
    # Sort: files with most class-0 content first (we never want to drop those),
    # then by minority ratio descending so balanced files are kept preferentially.
    file_stats.sort(key=lambda f: (f['c0'] == 0, -f['minority_ratio']))

    c1_budget  = int(total_c0 * target_ratio)   # max class-1 edges we want to keep
    kept_c0    = 0
    kept_c1    = 0
    keep_files = []
    drop_files = []

    for f in file_stats:
        if f['c0'] > 0:
            # Always keep files that contain any class-0 edges — they're rare
            keep_files.append(f)
            kept_c0 += f['c0']
            kept_c1 += f['c1']
        else:
            # Pure class-1 file — keep only if within budget
            if kept_c1 + f['c1'] <= c1_budget:
                keep_files.append(f)
                kept_c1 += f['c1']
            else:
                drop_files.append(f)

    if verbose:
        total_kept = kept_c0 + kept_c1
        print(f"\nAfter balancing: {len(keep_files)} files kept, {len(drop_files)} removed")
        print(f"  Class 0: {kept_c0:>12,}  ({kept_c0/total_kept*100:.1f}%)")
        print(f"  Class 1: {kept_c1:>12,}  ({kept_c1/total_kept*100:.1f}%)")
        print(f"  Ratio  : {kept_c1/kept_c0:.2f}:1")
        print(f"\n{'[DRY RUN] ' if dry_run else ''}{'Deleting' if not dry_run else 'Would delete'} {len(drop_files)} files")

    if not dry_run:
        for f in tqdm(drop_files, desc="Deleting", disable=not verbose):
            f['path'].unlink()

    return {
        'files_kept':    len(keep_files),
        'files_removed': len(drop_files),
        'kept_c0':       kept_c0,
        'kept_c1':       kept_c1,
        'orig_c0':       total_c0,
        'orig_c1':       total_c1,
    }


if __name__ == "__main__":
    edges_dir = Path("data/edges")

    for split in ['train', 'val', 'test']:
        if not (edges_dir / split).exists():
            continue
        print(f"\n{'='*60}\n{split.upper()}\n{'='*60}")
        balance_dataset_globally(
            edges_dir=edges_dir,
            split=split,
            target_ratio=1.0,   # aim for 1:1
            dry_run=False,        # flip to False when happy
            verbose=True,
        )