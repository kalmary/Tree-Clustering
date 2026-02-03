import numpy as np
from pathlib import Path
from typing import Union, Optional


def balance_edge_files(
    edges_dir: Union[str, Path],
    target_ratio: float = 1.0,
    split: str = 'train',
    label_col: int = -1,
    min_samples_per_file: int = 100,
    dry_run: bool = False,
    backup: bool = True
) -> dict:
    """
    Balance extremely imbalanced binary labeled data in .npy files.
    
    Strategy:
    1. Count labels across all files matching the split pattern
    2. Determine minority and majority classes
    3. Calculate target counts based on target_ratio
    4. Remove excess majority class samples from files
    5. Remove entire files if they become too small or contain only majority class
    
    Args:
        edges_dir: Directory containing .npy files
        target_ratio: Target ratio of majority:minority (1.0 = balanced, 2.0 = 2:1, etc.)
        split: Split identifier in filename (e.g., 'train', 'val', 'test')
        label_col: Column index containing binary labels (default: -1, last column)
        min_samples_per_file: Minimum samples to keep a file (files below this are removed)
        dry_run: If True, only report what would be done without modifying files
        backup: If True, create .bak copies before modifying files
    
    Returns:
        dict: Statistics about the balancing operation
    """
    edges_dir = Path(edges_dir.joinpath(split))
    
    if not edges_dir.exists():
        raise ValueError(f"Directory does not exist: {edges_dir}")
    
    # Find all .npy files matching the split
    pattern = f"*.npy"
    npy_files = sorted(edges_dir.glob(pattern))
    
    if not npy_files:
        print(f"WARNING: No .npy files found matching pattern '{pattern}' in {edges_dir}")
        return {}
    
    print(f"Found {len(npy_files)} files matching pattern '{pattern}'")
    
    # Phase 1: Analyze all files
    print("Phase 1: Analyzing label distribution...")
    file_stats = []
    total_counts = {0: 0, 1: 0}
    
    for file_path in npy_files:
        try:
            data = np.load(file_path)
            
            if data.ndim != 2 or data.shape[1] != 5:
                print(f"WARNING: Skipping {file_path.name}: expected shape (n, 5), got {data.shape}")
                continue
            
            labels = data[:, label_col].astype(int)
            unique, counts = np.unique(labels, return_counts=True)
            
            file_count = {int(u): int(c) for u, c in zip(unique, counts)}
            # Ensure both classes are represented in dict
            file_count.setdefault(0, 0)
            file_count.setdefault(1, 0)
            
            file_stats.append({
                'path': file_path,
                'data': data,
                'labels': labels,
                'counts': file_count,
                'total': len(labels)
            })
            
            total_counts[0] += file_count[0]
            total_counts[1] += file_count[1]
            
        except Exception as e:
            print(f"ERROR: Error reading {file_path.name}: {e}")
            continue
    
    if not file_stats:
        print("ERROR: No valid files to process")
        return {}
    
    # Determine minority and majority classes
    minority_class = 0 if total_counts[0] < total_counts[1] else 1
    majority_class = 1 - minority_class
    
    minority_count = total_counts[minority_class]
    majority_count = total_counts[majority_class]
    
    print(f"\nOriginal distribution:")
    print(f"  Class {minority_class} (minority): {minority_count:,} samples")
    print(f"  Class {majority_class} (majority): {majority_count:,} samples")
    print(f"  Imbalance ratio: {majority_count/max(minority_count, 1):.2f}:1")
    
    # Phase 2: Calculate target counts
    target_majority_count = int(minority_count * target_ratio)
    samples_to_remove = majority_count - target_majority_count
    
    print(f"\nTarget distribution (ratio {target_ratio}:1):")
    print(f"  Class {minority_class}: {minority_count:,} samples (unchanged)")
    print(f"  Class {majority_class}: {target_majority_count:,} samples")
    print(f"  Samples to remove: {samples_to_remove:,}")
    
    if samples_to_remove <= 0:
        print("Data is already balanced or minority class is larger. No action needed.")
        return {
            'files_processed': 0,
            'files_removed': 0,
            'samples_removed': 0,
            'original_counts': total_counts,
            'final_counts': total_counts
        }
    
    # Phase 3: Balance files
    print(f"\nPhase 2: {'Simulating' if dry_run else 'Balancing'} files...")
    
    files_modified = 0
    files_removed = 0
    total_removed = 0
    removed_per_file = []
    
    for file_info in file_stats:
        file_path = file_info['path']
        data = file_info['data']
        labels = file_info['labels']
        counts = file_info['counts']
        
        majority_in_file = counts[majority_class]
        minority_in_file = counts[minority_class]
        
        if majority_in_file == 0:
            # File contains only minority class - keep as is
            continue
        
        # Calculate how many majority samples to keep in this file
        # Proportional to the file's contribution to total majority samples
        proportion = majority_in_file / majority_count
        target_majority_in_file = int(target_majority_count * proportion)
        
        # Ensure we keep at least some samples if there's minority class
        if minority_in_file > 0:
            target_majority_in_file = max(target_majority_in_file, 1)
        
        samples_to_remove_from_file = majority_in_file - target_majority_in_file
        
        if samples_to_remove_from_file <= 0:
            continue
        
        # Get indices of majority and minority samples
        majority_indices = np.where(labels == majority_class)[0]
        minority_indices = np.where(labels == minority_class)[0]
        
        # Randomly select majority samples to keep
        np.random.seed(42)  # For reproducibility
        keep_majority_indices = np.random.choice(
            majority_indices, 
            size=target_majority_in_file, 
            replace=False
        )
        
        # Combine minority and selected majority samples
        keep_indices = np.concatenate([minority_indices, keep_majority_indices])
        keep_indices = np.sort(keep_indices)
        
        new_data = data[keep_indices]
        new_total = len(new_data)
        
        # Check if file should be removed
        if new_total < min_samples_per_file:
            print(f"  {file_path.name}: {new_total} samples < {min_samples_per_file} minimum, REMOVING file")
            if not dry_run:
                if backup:
                    backup_path = file_path.with_suffix('.npy.bak')
                    file_path.rename(backup_path)
                else:
                    file_path.unlink()
            files_removed += 1
            total_removed += majority_in_file
        else:
            print(f"  {file_path.name}: {counts[0]}/{counts[1]} -> "
                       f"{np.sum(new_data[:, label_col] == 0)}/{np.sum(new_data[:, label_col] == 1)} "
                       f"(removed {samples_to_remove_from_file} samples)")
            
            if not dry_run:
                if backup:
                    backup_path = file_path.with_suffix('.npy.bak')
                    np.save(backup_path, data)
                
                np.save(file_path, new_data)
            
            files_modified += 1
            total_removed += samples_to_remove_from_file
            removed_per_file.append(samples_to_remove_from_file)
    
    # Final statistics
    final_majority = majority_count - total_removed
    final_counts = {
        minority_class: minority_count,
        majority_class: final_majority
    }
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Files modified: {files_modified}")
    print(f"  Files removed: {files_removed}")
    print(f"  Total samples removed: {total_removed:,}")
    print(f"\nFinal distribution:")
    print(f"  Class {minority_class}: {minority_count:,}")
    print(f"  Class {majority_class}: {final_majority:,}")
    print(f"  Final ratio: {final_majority/max(minority_count, 1):.2f}:1")
    
    if dry_run:
        print("\nThis was a dry run. No files were modified.")
    
    return {
        'files_processed': len(file_stats),
        'files_modified': files_modified,
        'files_removed': files_removed,
        'samples_removed': total_removed,
        'original_counts': total_counts,
        'final_counts': final_counts,
        'removed_per_file': removed_per_file
    }


if __name__ == "__main__":
    # Example usage
    edges_dir = Path("data/edges")
    
    # First do a dry run to see what would happen
    print("=" * 80)
    print("DRY RUN - No files will be modified")
    print("=" * 80)
    stats = balance_edge_files(
        edges_dir, 
        target_ratio=1.0,  # 1:1 ratio
        split='train',
        dry_run=True
    )
    
    # Uncomment to actually perform the balancing
    print("\n" + "=" * 80)
    print("ACTUAL RUN - Files will be modified (backups created)")
    print("=" * 80)
    stats = balance_edge_files(
        edges_dir, 
        target_ratio=1.0,
        split='train',
        dry_run=False,
        backup=False
    )