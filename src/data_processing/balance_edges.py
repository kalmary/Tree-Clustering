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
    min_minority_ratio: float = 0.20  # Keep files with ≥20% minority class
) -> dict:
    """
    Balance graph files by removing those with extreme class imbalance.
    Removes files where minority class is <20% of edges (i.e., >80% single class).
    
    Args:
        edges_dir: Directory containing .pt files
        split: Split to process
        dry_run: Only report, don't modify
        backup: Create .bak copies
        verbose: Show progress
        min_minority_ratio: Minimum ratio of minority class to keep file (0.20 = 20%)
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
        print(f"Removing files with <{min_minority_ratio*100:.0f}% minority class")
    
    # Scan and categorize files
    if verbose:
        print(f"\nScanning files for class distribution...")
    
    keep_files = []
    remove_files = []
    
    stats = {
        'files_processed': 0,
        'files_kept': 0,
        'files_removed': 0,
        'original_counts': defaultdict(int),
        'final_counts': defaultdict(int),
        'imbalance_histogram': defaultdict(int)
    }
    
    pbar = tqdm(pt_files, desc="Scanning files", disable=not verbose)
    
    for file_path in pbar:
        try:
            graph = torch.load(file_path, map_location='cpu', weights_only=True)
            
            labels = graph['y']
            class_0 = (labels == 0).sum().item()
            class_1 = (labels == 1).sum().item()
            total = class_0 + class_1
            
            stats['original_counts'][0] += class_0
            stats['original_counts'][1] += class_1
            stats['files_processed'] += 1
            
            # Calculate minority ratio
            minority_count = min(class_0, class_1)
            minority_ratio = minority_count / total if total > 0 else 0
            
            # Bin for histogram (0-10%, 10-20%, etc.)
            bin_idx = int(minority_ratio * 10) * 10
            stats['imbalance_histogram'][bin_idx] += 1
            
            file_info = {
                'path': file_path,
                'class_0': class_0,
                'class_1': class_1,
                'total': total,
                'minority_ratio': minority_ratio
            }
            
            # Keep if minority class is ≥ threshold
            if minority_ratio >= min_minority_ratio:
                keep_files.append(file_info)
                stats['files_kept'] += 1
                stats['final_counts'][0] += class_0
                stats['final_counts'][1] += class_1
            else:
                remove_files.append(file_info)
                stats['files_removed'] += 1
            
            del graph
            
        except Exception as e:
            if verbose:
                pbar.write(f"Error processing {file_path.name}: {e}")
            continue
    
    if verbose:
        print(f"\nResults:")
        print(f"  Files to keep: {len(keep_files)} ({len(keep_files)/len(pt_files)*100:.1f}%)")
        print(f"  Files to remove: {len(remove_files)} ({len(remove_files)/len(pt_files)*100:.1f}%)")
        
        print(f"\nMinority class ratio distribution:")
        for bin_val in sorted(stats['imbalance_histogram'].keys()):
            count = stats['imbalance_histogram'][bin_val]
            pct = count / stats['files_processed'] * 100
            bar = '█' * int(pct / 2)
            print(f"  {bin_val:3d}-{bin_val+10:3d}%: {count:5d} files ({pct:5.1f}%) {bar}")
    
    # Remove files
    if not dry_run and remove_files:
        if verbose:
            print(f"\nRemoving {len(remove_files)} files...")
        
        pbar = tqdm(remove_files, desc="Removing files", disable=not verbose)
        
        for file_info in pbar:
            file_path = file_info['path']
            
            if backup:
                file_path.rename(file_path.with_suffix('.pt.bak'))
            else:
                file_path.unlink()
    
    # Summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        print(f"{'='*80}\n")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Files kept: {stats['files_kept']} ({stats['files_kept']/stats['files_processed']*100:.1f}%)")
        print(f"Files removed: {stats['files_removed']} ({stats['files_removed']/stats['files_processed']*100:.1f}%)")
        
        print(f"\nOriginal distribution:")
        print(f"  Class 0: {stats['original_counts'][0]:,}")
        print(f"  Class 1: {stats['original_counts'][1]:,}")
        total_orig = stats['original_counts'][0] + stats['original_counts'][1]
        print(f"  Class 0: {stats['original_counts'][0]/total_orig*100:.1f}%")
        print(f"  Class 1: {stats['original_counts'][1]/total_orig*100:.1f}%")
        if stats['original_counts'][1] > 0:
            print(f"  Ratio: {stats['original_counts'][0]/stats['original_counts'][1]:.2f}:1")
        
        print(f"\nFinal distribution:")
        print(f"  Class 0: {stats['final_counts'][0]:,}")
        print(f"  Class 1: {stats['final_counts'][1]:,}")
        total_final = stats['final_counts'][0] + stats['final_counts'][1]
        if total_final > 0:
            print(f"  Class 0: {stats['final_counts'][0]/total_final*100:.1f}%")
            print(f"  Class 1: {stats['final_counts'][1]/total_final*100:.1f}%")
            if stats['final_counts'][1] > 0:
                print(f"  Ratio: {stats['final_counts'][0]/stats['final_counts'][1]:.2f}:1")
        
        print(f"\nImprovement:")
        if stats['original_counts'][1] > 0 and stats['final_counts'][1] > 0:
            orig_ratio = stats['original_counts'][0]/stats['original_counts'][1]
            final_ratio = stats['final_counts'][0]/stats['final_counts'][1]
            improvement = (orig_ratio - final_ratio) / orig_ratio * 100
            print(f"  Imbalance reduced by {improvement:.1f}%")
        
        if dry_run:
            print("\nDry run - no files modified")
    
    return stats


if __name__ == "__main__":
    edges_dir = Path("data/edges")
    
    # First, run dry run to see what would happen
    # print("DRY RUN - checking what would be removed:\n")
    # for split in ['train', 'val', 'test']:
    #     if (edges_dir / split).exists():
    #         balance_graph_files_by_edge_ratio(
    #             edges_dir,
    #             split=split,
    #             dry_run=True,
    #             min_minority_ratio=0.20
    #         )
    
    # Uncomment to actually remove files
    print("\n\nACTUAL RUN - removing files:\n")
    for split in ['train', 'val', 'test']:
        if (edges_dir / split).exists():
            balance_graph_files_by_edge_ratio(
                edges_dir,
                split=split,
                dry_run=False,
                backup=False,
                min_minority_ratio=0.2
            )