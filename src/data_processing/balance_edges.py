import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple


def check_file_variance(
    file_path: Path,
    label_col: int = -1,
    imbalance_threshold: float = 0.95
) -> dict:
    """
    Check if a single file has class variance issues.
    
    Args:
        file_path: Path to .npy file
        label_col: Column index containing labels
        imbalance_threshold: Threshold for considering a file highly imbalanced
    
    Returns:
        dict with file statistics and flags
    """
    try:
        data = np.load(file_path)
        
        if data.ndim != 2 or data.shape[1] != 9:
            return {
                'valid': False,
                'error': f'Invalid shape: expected (n, 9), got {data.shape}'
            }
        
        labels = data[:, label_col].astype(int)
        n_samples = len(labels)
        
        class_0_count = np.sum(labels == 0)
        class_1_count = np.sum(labels == 1)
        
        class_0_pct = class_0_count / n_samples if n_samples > 0 else 0
        class_1_pct = class_1_count / n_samples if n_samples > 0 else 0
        
        return {
            'valid': True,
            'path': file_path,
            'data': data,
            'labels': labels,
            'n_samples': n_samples,
            'class_0_count': class_0_count,
            'class_1_count': class_1_count,
            'class_0_pct': class_0_pct,
            'class_1_pct': class_1_pct,
            'only_class_0': class_1_count == 0,
            'only_class_1': class_0_count == 0,
            'highly_imbalanced': class_0_pct > imbalance_threshold or class_1_pct > imbalance_threshold,
            'single_class': class_0_count == 0 or class_1_count == 0
        }
    
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def merge_single_class_files(
    single_class_files: List[dict],
    target_class: int,
    max_samples_per_merged: int = 5000
) -> List[np.ndarray]:
    """
    Merge multiple single-class files into larger mixed batches.
    
    Args:
        single_class_files: List of file info dicts for single-class files
        target_class: The class these files contain (0 or 1)
        max_samples_per_merged: Maximum samples per merged file
    
    Returns:
        List of merged data arrays
    """
    merged_files = []
    current_batch = []
    current_size = 0
    
    for file_info in single_class_files:
        data = file_info['data']
        
        if current_size + len(data) <= max_samples_per_merged:
            current_batch.append(data)
            current_size += len(data)
        else:
            if current_batch:
                merged_files.append(np.vstack(current_batch))
            current_batch = [data]
            current_size = len(data)
    
    # Add remaining batch
    if current_batch:
        merged_files.append(np.vstack(current_batch))
    
    return merged_files


def balance_edge_files(
    edges_dir: Union[str, Path],
    target_ratio: float = 1.0,
    split: str = 'train',
    label_col: int = -1,
    min_samples_per_file: int = 100,
    dry_run: bool = False,
    backup: bool = True,
    verbose: bool = True,
    handle_single_class: str = 'remove',  # 'remove', 'merge', or 'keep'
    imbalance_threshold: float = 0.95
) -> dict:
    """
    Balance extremely imbalanced binary labeled data in .npy files.
    
    Strategy:
    1. Check for class variance issues (single-class files, highly imbalanced)
    2. Handle problematic files based on strategy
    3. Count labels across all remaining files
    4. Determine minority and majority classes
    5. Calculate target counts based on target_ratio
    6. Remove excess majority class samples from files
    7. Remove entire files if they become too small
    
    Args:
        edges_dir: Directory containing .npy files
        target_ratio: Target ratio of majority:minority (1.0 = balanced, 2.0 = 2:1, etc.)
        split: Split identifier in filename (e.g., 'train', 'val', 'test')
        label_col: Column index containing binary labels (default: -1, last column)
        min_samples_per_file: Minimum samples to keep a file (files below this are removed)
        dry_run: If True, only report what would be done without modifying files
        backup: If True, create .bak copies before modifying files
        verbose: If True, print progress and statistics
        handle_single_class: How to handle single-class files:
            - 'remove': Delete files containing only one class
            - 'merge': Merge single-class files together (creates larger mixed files)
            - 'keep': Keep them as-is (not recommended)
        imbalance_threshold: Files with >this fraction of one class are flagged (0.95 = 95%)
    
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
        if verbose:
            print(f"WARNING: No .npy files found matching pattern '{pattern}' in {edges_dir}")
        return {}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"BALANCE EDGE FILES - {split.upper()} SPLIT")
        print(f"{'='*80}\n")
        print(f"Found {len(npy_files)} files matching pattern '{pattern}'")
    
    # Phase 0: Check for class variance issues
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 0: Checking for class variance issues...")
        print(f"{'='*80}\n")
    
    file_checks = []
    only_class_0_files = []
    only_class_1_files = []
    highly_imbalanced_files = []
    valid_files = []
    
    for file_path in npy_files:
        check_result = check_file_variance(file_path, label_col, imbalance_threshold)
        
        if not check_result['valid']:
            if verbose:
                print(f"WARNING: Skipping {file_path.name}: {check_result['error']}")
            continue
        
        file_checks.append(check_result)
        
        if check_result['only_class_0']:
            only_class_0_files.append(check_result)
        elif check_result['only_class_1']:
            only_class_1_files.append(check_result)
        elif check_result['highly_imbalanced']:
            highly_imbalanced_files.append(check_result)
        else:
            valid_files.append(check_result)
    
    # Report variance issues
    if verbose:
        total_single_class = len(only_class_0_files) + len(only_class_1_files)
        
        if total_single_class > 0 or highly_imbalanced_files:
            print(f"⚠️  CLASS VARIANCE ISSUES DETECTED:")
            print(f"   Files with only class 0: {len(only_class_0_files)}")
            print(f"   Files with only class 1: {len(only_class_1_files)}")
            print(f"   Highly imbalanced files (>{imbalance_threshold*100:.0f}% one class): {len(highly_imbalanced_files)}")
            print(f"   Reasonably balanced files: {len(valid_files)}")
            print(f"\n   Strategy for single-class files: {handle_single_class.upper()}")
        else:
            print(f"✓ No class variance issues detected")
            print(f"  All {len(valid_files)} files have reasonable class balance")
    
    # Handle single-class files
    single_class_files_removed = 0
    single_class_files_merged = 0
    
    if handle_single_class == 'remove' and (only_class_0_files or only_class_1_files):
        if verbose:
            print(f"\nRemoving {len(only_class_0_files) + len(only_class_1_files)} single-class files...")
        
        for file_info in only_class_0_files + only_class_1_files:
            if verbose:
                print(f"  Removing {file_info['path'].name} ({file_info['n_samples']} samples, only class {0 if file_info['only_class_0'] else 1})")
            
            if not dry_run:
                if backup:
                    backup_path = file_info['path'].with_suffix('.npy.bak')
                    file_info['path'].rename(backup_path)
                else:
                    file_info['path'].unlink()
            
            single_class_files_removed += 1
        
        # Remove from file_checks
        file_checks = [f for f in file_checks if not f['single_class']]
    
    elif handle_single_class == 'merge' and (only_class_0_files or only_class_1_files):
        if verbose:
            print(f"\nMerging single-class files...")
        
        # Note: Merging strategy creates larger files but doesn't mix classes
        # This is mainly to consolidate storage, not to create balanced files
        if only_class_0_files:
            merged_0 = merge_single_class_files(only_class_0_files, 0)
            if verbose:
                print(f"  Merged {len(only_class_0_files)} class-0 files into {len(merged_0)} larger files")
        
        if only_class_1_files:
            merged_1 = merge_single_class_files(only_class_1_files, 1)
            if verbose:
                print(f"  Merged {len(only_class_1_files)} class-1 files into {len(merged_1)} larger files")
        
        # Save merged files and remove originals
        if not dry_run:
            merged_count = 0
            
            if only_class_0_files:
                for i, merged_data in enumerate(merged_0):
                    new_path = edges_dir / f"merged_class0_{i:04d}.npy"
                    np.save(new_path, merged_data)
                    merged_count += 1
                
                for file_info in only_class_0_files:
                    if backup:
                        backup_path = file_info['path'].with_suffix('.npy.bak')
                        file_info['path'].rename(backup_path)
                    else:
                        file_info['path'].unlink()
            
            if only_class_1_files:
                for i, merged_data in enumerate(merged_1):
                    new_path = edges_dir / f"merged_class1_{i:04d}.npy"
                    np.save(new_path, merged_data)
                    merged_count += 1
                
                for file_info in only_class_1_files:
                    if backup:
                        backup_path = file_info['path'].with_suffix('.npy.bak')
                        file_info['path'].rename(backup_path)
                    else:
                        file_info['path'].unlink()
            
            single_class_files_merged = len(only_class_0_files) + len(only_class_1_files)
            
            if verbose:
                print(f"  Created {merged_count} merged files")
        
        # Note: We keep single-class files in file_checks for balancing
        # but they're now merged into larger files
    
    if not file_checks:
        if verbose:
            print("ERROR: No valid files to process after variance check")
        return {}
    
    # Phase 1: Analyze all files
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 1: Analyzing label distribution...")
        print(f"{'='*80}\n")
    
    file_stats = []
    total_counts = {0: 0, 1: 0}
    
    for check_result in file_checks:
        file_count = {
            0: check_result['class_0_count'],
            1: check_result['class_1_count']
        }
        
        file_stats.append({
            'path': check_result['path'],
            'data': check_result['data'],
            'labels': check_result['labels'],
            'counts': file_count,
            'total': check_result['n_samples']
        })
        
        total_counts[0] += file_count[0]
        total_counts[1] += file_count[1]
    
    # Determine minority and majority classes
    minority_class = 0 if total_counts[0] < total_counts[1] else 1
    majority_class = 1 - minority_class
    
    minority_count = total_counts[minority_class]
    majority_count = total_counts[majority_class]
    
    if verbose:
        print(f"Original distribution:")
        print(f"  Class {minority_class} (minority): {minority_count:,} samples")
        print(f"  Class {majority_class} (majority): {majority_count:,} samples")
        print(f"  Imbalance ratio: {majority_count/max(minority_count, 1):.2f}:1")
    
    # Phase 2: Calculate target counts
    target_majority_count = int(minority_count * target_ratio)
    samples_to_remove = majority_count - target_majority_count
    
    if verbose:
        print(f"\nTarget distribution (ratio {target_ratio}:1):")
        print(f"  Class {minority_class}: {minority_count:,} samples (unchanged)")
        print(f"  Class {majority_class}: {target_majority_count:,} samples")
        print(f"  Samples to remove: {samples_to_remove:,}")
    
    if samples_to_remove <= 0:
        if verbose:
            print("\nData is already balanced or minority class is larger. No action needed.")
        return {
            'files_processed': len(file_stats),
            'files_removed': single_class_files_removed,
            'files_merged': single_class_files_merged,
            'samples_removed': 0,
            'original_counts': total_counts,
            'final_counts': total_counts,
            'single_class_files_removed': single_class_files_removed,
            'single_class_files_merged': single_class_files_merged
        }
    
    # Phase 3: Balance files
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 2: {'Simulating' if dry_run else 'Balancing'} files...")
        print(f"{'='*80}\n")
    
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
            if verbose:
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
            if verbose:
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
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        print(f"{'='*80}\n")
        print(f"Single-class files removed: {single_class_files_removed}")
        print(f"Single-class files merged: {single_class_files_merged}")
        print(f"Files modified: {files_modified}")
        print(f"Files removed (too small): {files_removed}")
        print(f"Total samples removed: {total_removed:,}")
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
        'removed_per_file': removed_per_file,
        'single_class_files_removed': single_class_files_removed,
        'single_class_files_merged': single_class_files_merged,
        'highly_imbalanced_count': len(highly_imbalanced_files)
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
        dry_run=True,
        verbose=True,
        handle_single_class='remove',  # or 'merge' or 'keep'
        backup=False,
        imbalance_threshold=0.7
    )
    
    # Uncomment to actually perform the balancing
    print("\n" + "=" * 80)
    print("ACTUAL RUN - Files will be modified")
    print("=" * 80)
    stats = balance_edge_files(
        edges_dir, 
        target_ratio=1.0,
        split='train',
        dry_run=False,
        backup=False,
        verbose=True,
        handle_single_class='remove'  # Remove single-class files
    )