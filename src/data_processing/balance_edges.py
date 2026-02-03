import numpy as np
from pathlib import Path
from tqdm import tqdm


def balance_edge_files(edges_dir: Path, target_ratio: float = 3.0, split: str = 'train'):
    """
    Balance edge data by undersampling majority class (class 1 - same tree).
    
    Args:
        edges_dir: Path to edges directory
        target_ratio: Desired class_1/class_0 ratio (e.g., 3.0 means 3:1)
        split: Which split to balance ('train', 'val', 'test')
    """
    split_dir = edges_dir / split
    npy_files = sorted(split_dir.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {split_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"BALANCING EDGE DATA - {split.upper()} SPLIT")
    print(f"Target ratio (class_1:class_0): {target_ratio}:1")
    print(f"{'='*60}\n")
    
    total_before_0 = 0
    total_before_1 = 0
    total_after_0 = 0
    total_after_1 = 0
    
    for npy_file in tqdm(npy_files, desc="Balancing files"):
        data = np.load(npy_file)
        
        if data.shape[0] == 0:
            continue
        
        X = data[:, :-1]
        y = data[:, -1]
        
        # Count classes
        class_0_mask = y == 0
        class_1_mask = y == 1
        
        n_class_0 = class_0_mask.sum()
        n_class_1 = class_1_mask.sum()
        
        total_before_0 += n_class_0
        total_before_1 += n_class_1
        
        if n_class_0 == 0 or n_class_1 == 0:
            # Skip files with only one class
            total_after_0 += n_class_0
            total_after_1 += n_class_1
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
    
    # Print summary
    print(f"\n{'='*60}")
    print("BALANCING SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nBEFORE:")
    print(f"  Class 0: {total_before_0:,} samples ({total_before_0/(total_before_0+total_before_1)*100:.2f}%)")
    print(f"  Class 1: {total_before_1:,} samples ({total_before_1/(total_before_0+total_before_1)*100:.2f}%)")
    print(f"  Ratio: {total_before_1/total_before_0:.2f}:1")
    
    print(f"\nAFTER:")
    print(f"  Class 0: {total_after_0:,} samples ({total_after_0/(total_after_0+total_after_1)*100:.2f}%)")
    print(f"  Class 1: {total_after_1:,} samples ({total_after_1/(total_after_0+total_after_1)*100:.2f}%)")
    print(f"  Ratio: {total_after_1/total_after_0:.2f}:1")
    
    print(f"\nReduction: {(1 - (total_after_0+total_after_1)/(total_before_0+total_before_1))*100:.1f}% of data removed")
    print()


def main():
    edges_dir = Path("data/edges")
    
    # Balance only training set (keep val/test as-is for fair evaluation)
    balance_edge_files(edges_dir, target_ratio=2., split='train')


if __name__ == "__main__":
    main()
