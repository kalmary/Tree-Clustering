import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import random


def balance_graph_files_streaming(
    edges_dir: Path,
    split: str = 'train',
    dry_run: bool = False,
    backup: bool = False,
    verbose: bool = True,
    single_class_keep_ratio: float = 0.05  # Keep 5%, remove 95%
) -> dict:
    """
    Balance graph files using streaming approach.
    Remove 95% of single-class graphs, keep 5% for diversity.
    
    Args:
        edges_dir: Directory containing .pt files
        split: Split to process
        dry_run: Only report, don't modify
        backup: Create .bak copies
        verbose: Show progress
        single_class_keep_ratio: Fraction of single-class graphs to keep (0.05 = 5%)
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
        print(f"BALANCE GRAPH FILES - {split.upper()} (STREAMING)")
        print(f"{'='*80}\n")
        print(f"Found {len(pt_files)} files")
    
    # Phase 1: Identify single-class files
    if verbose:
        print(f"\nPhase 1: Identifying single-class files...")
    
    single_class_files = []
    mixed_class_files = []
    
    stats = {
        'files_processed': 0,
        'files_removed': 0,
        'single_class_total': 0,
        'single_class_kept': 0,
        'single_class_removed': 0,
        'original_counts': defaultdict(int),
        'final_counts': defaultdict(int)
    }
    
    pbar = tqdm(pt_files, desc="Scanning files", disable=not verbose)
    
    for file_path in pbar:
        try:
            graph = torch.load(file_path, map_location='cpu', weights_only=True)
            
            labels = graph['y']
            class_0 = (labels == 0).sum().item()
            class_1 = (labels == 1).sum().item()
            
            stats['original_counts'][0] += class_0
            stats['original_counts'][1] += class_1
            stats['files_processed'] += 1
            
            # Check if single-class (100% one class)
            is_single_class = (class_0 == 0 or class_1 == 0)
            
            if is_single_class:
                single_class_files.append({
                    'path': file_path,
                    'class_0': class_0,
                    'class_1': class_1
                })
                stats['single_class_total'] += 1
            else:
                mixed_class_files.append({
                    'path': file_path,
                    'class_0': class_0,
                    'class_1': class_1
                })
            
            del graph
            
        except Exception as e:
            if verbose:
                pbar.write(f"Error processing {file_path.name}: {e}")
            continue
    
    if verbose:
        print(f"\nFound:")
        print(f"  Single-class files: {len(single_class_files)}")
        print(f"  Mixed-class files: {len(mixed_class_files)}")
    
    # Phase 2: Randomly select which single-class files to keep
    if verbose:
        print(f"\nPhase 2: Removing {(1-single_class_keep_ratio)*100:.0f}% of single-class files...")
    
    random.seed(42)  # Reproducibility
    n_keep = int(len(single_class_files) * single_class_keep_ratio)
    
    random.shuffle(single_class_files)
    keep_single_class = single_class_files[:n_keep]
    remove_single_class = single_class_files[n_keep:]
    
    stats['single_class_kept'] = len(keep_single_class)
    stats['single_class_removed'] = len(remove_single_class)
    
    if verbose:
        print(f"  Keeping: {len(keep_single_class)} single-class files")
        print(f"  Removing: {len(remove_single_class)} single-class files")
    
    # Phase 3: Remove files
    pbar = tqdm(remove_single_class, desc="Removing files", disable=not verbose)
    
    for file_info in pbar:
        file_path = file_info['path']
        
        if not dry_run:
            if backup:
                file_path.rename(file_path.with_suffix('.pt.bak'))
            else:
                file_path.unlink()
        
        stats['files_removed'] += 1
    
    # Calculate final counts
    for file_info in mixed_class_files + keep_single_class:
        stats['final_counts'][0] += file_info['class_0']
        stats['final_counts'][1] += file_info['class_1']
    
    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        print(f"{'='*80}\n")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files removed: {stats['files_removed']} ({stats['files_removed']/stats['files_processed']*100:.1f}%)")
        print(f"\nSingle-class handling:")
        print(f"  Total: {stats['single_class_total']}")
        print(f"  Kept: {stats['single_class_kept']} ({single_class_keep_ratio*100:.0f}%)")
        print(f"  Removed: {stats['single_class_removed']} ({(1-single_class_keep_ratio)*100:.0f}%)")
        print(f"\nOriginal distribution:")
        print(f"  Class 0: {stats['original_counts'][0]:,}")
        print(f"  Class 1: {stats['original_counts'][1]:,}")
        if stats['original_counts'][1] > 0:
            print(f"  Ratio: {stats['original_counts'][0]/stats['original_counts'][1]:.2f}:1")
        print(f"\nFinal distribution:")
        print(f"  Class 0: {stats['final_counts'][0]:,}")
        print(f"  Class 1: {stats['final_counts'][1]:,}")
        if stats['final_counts'][1] > 0:
            print(f"  Ratio: {stats['final_counts'][0]/stats['final_counts'][1]:.2f}:1")
        
        if dry_run:
            print("\nDry run - no files modified")
    
    return stats


if __name__ == "__main__":
    edges_dir = Path("data/edges")
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split}...")
        stats = balance_graph_files_streaming(
            edges_dir,
            split=split,
            dry_run=False,
            backup=False,
            verbose=True,
            single_class_keep_ratio=0.05  # Keep 5%, remove 95%
        )