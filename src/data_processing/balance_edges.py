import numpy as np
from pathlib import Path
from tqdm import tqdm


def balance_edge_files(edges_dir: Path, target_ratio: float = 1.0, split: str = 'train', 
                       min_class_0: int = 50, remove_extreme: bool = True):
    """
    Balance edge data by undersampling majority class (class 1 - same tree).
    
    Args:
        edges_dir: Path to edges directory
        target_ratio: Desired class_1/class_0 ratio (e.g., 1.0 means 1:1)
        split: Which split to balance ('train', 'val', 'test')
        min_class_0: Minimum class 0 samples required, else remove file
        remove_extreme: Remove files with extreme imbalance (>20:1)
    """
    split_dir = edges_dir / split
    npy_files = sorted(split_dir.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {split_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"BALANCING EDGE DATA - {split.upper()} SPLIT")
    print(f"Target ratio (class_1:class_0): {target_ratio}:1")
    print(f"Min class 0 samples: {min_class_0}")
    print(f"Remove extreme imbalance: {remove_extreme}")
    print(f"{'='*60}\n")
    
    total_before_0 = 0
    total_before_1 = 0
    total_after_0 = 0
    total_after_1 = 0
    files_removed = 0
    files_balanced = 0
    
    for npy_file in tqdm(npy_files, desc="Balancing files"):
        data = np.load(npy_file)
        
        if data.shape[0] == 0:
            npy_file.unlink()
            files_removed += 1
            continue
        
        y = data[:, -1]
        
        # Count classes
        class_0_mask = y == 0
        class_1_mask = y == 1
        
        n_class_0 = class_0_mask.sum()
        n_class_1 = class_1_mask.sum()
        
        total_before_0 += n_class_0
        total_before_1 += n_class_1
        
        # Remove files with insufficient class 0 samples
        if n_class_0 < min_class_0:
            npy_file.unlink()
            files_removed += 1
            continue
        
        # Remove files with only one class
        if n_class_0 == 0 or n_class_1 == 0:
            npy_file.unlink()
            files_removed += 1
            continue
        
        # Remove files with extreme imbalance
        current_ratio = n_class_1 / n_class_0
        if remove_extreme and current_ratio > 20:
            npy_file.unlink()
            files_removed += 1
            continue
        
        # Calculate target number of class 1 samples
        target_class_1 = int(n_class_0 * target_ratio)
        
        if n_class_1 <= target_class_1:
            # Already balanced enough
            total_after_0 += n_class_0
            total_after_1 += n_class_1
            continue
        
        # Undersample class 1
        class_0_indices = np.where(class_0_mask)[0]
        class_1_indices = np.where(class_1_mask)[0]
        
        # Randomly sample from class 1
        sampled_class_1_indices = np.random.choice(
            class_1_indices, 
            size=target_class_1, 
            replace=False
        )
        
        # Combine indices
        balanced_indices = np.concatenate([class_0_indices, sampled_class_1_indices])
        np.random.shuffle(balanced_indices)
        
        # Create balanced data
        balanced_data = data[balanced_indices]
        
        # Save
        np.save(npy_file, balanced_data)
        
        total_after_0 += n_class_0
        total_after_1 += target_class_1
        files_balanced += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("BALANCING SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nFILES:")
    print(f"  Total processed: {len(npy_files)}")
    print(f"  Files removed: {files_removed}")
    print(f"  Files balanced: {files_balanced}")
    print(f"  Files remaining: {len(npy_files) - files_removed}")
    
    if total_before_0 + total_before_1 > 0:
        print(f"\nBEFORE:")
        print(f"  Class 0 (different trees): {total_before_0:,} ({total_before_0/(total_before_0+total_before_1)*100:.2f}%)")
        print(f"  Class 1 (same tree): {total_before_1:,} ({total_before_1/(total_before_0+total_before_1)*100:.2f}%)")
        print(f"  Ratio (1:0): {total_before_1/total_before_0:.2f}:1")
    
    if total_after_0 + total_after_1 > 0:
        print(f"\nAFTER:")
        print(f"  Class 0 (different trees): {total_after_0:,} ({total_after_0/(total_after_0+total_after_1)*100:.2f}%)")
        print(f"  Class 1 (same tree): {total_after_1:,} ({total_after_1/(total_after_0+total_after_1)*100:.2f}%)")
        print(f"  Ratio (1:0): {total_after_1/total_after_0:.2f}:1")
        
        print(f"\nREDUCTION: {(1 - (total_after_0+total_after_1)/(total_before_0+total_before_1))*100:.1f}% of data removed")
    print()


def main():
    edges_dir = Path("data/edges")
    
    # Balance only training set (keep val/test as-is for fair evaluation)
    balance_edge_files(
        edges_dir, 
        target_ratio=1.0,  # 1:1 ratio
        split='train',
        min_class_0=50,  # Remove files with <50 different-tree edges
        remove_extreme=True  # Remove files with >20:1 imbalance
    )


if __name__ == "__main__":
    main()
