import numpy as np
from pathlib import Path
from tqdm import tqdm


def balance_edge_dataset(input_dir, output_dir, target_ratio=0.4, verbose=True):
    """
    Balance edge dataset by undersampling majority class.
    
    Args:
        input_dir: Directory with preprocessed edge files (N_edges, 5)
        output_dir: Directory to save balanced edge files
        target_ratio: Target ratio of negative samples (class 0), default 0.4 means 40% class 0, 60% class 1
        verbose: Print statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = sorted(input_path.rglob("*.npy"))
    
    if verbose:
        print(f"\n=== Balancing dataset ===")
        print(f"Found {len(npy_files)} files")
        print(f"Target ratio: {target_ratio:.1%} class 0, {1-target_ratio:.1%} class 1")
    
    total_original = 0
    total_balanced = 0
    
    for npy_file in tqdm(npy_files, desc="Balancing files"):
        edges = np.load(npy_file)
        
        if len(edges) == 0:
            # Save empty file
            relative_path = npy_file.relative_to(input_path)
            out_file = output_path / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_file, edges)
            continue
        
        # Split by class
        labels = edges[:, 4]
        class_0_mask = labels == 0
        class_1_mask = labels == 1
        
        edges_0 = edges[class_0_mask]
        edges_1 = edges[class_1_mask]
        
        n_class_0 = len(edges_0)
        n_class_1 = len(edges_1)
        
        total_original += len(edges)
        
        # Calculate how many samples we need from each class
        # target_ratio = n_0 / (n_0 + n_1)
        # We keep all of minority class and undersample majority
        
        if n_class_1 > n_class_0:
            # Class 1 is majority - undersample it
            n_keep_1 = int(n_class_0 * (1 - target_ratio) / target_ratio)
            n_keep_1 = min(n_keep_1, n_class_1)  # Don't exceed available samples
            
            indices_1 = np.random.choice(n_class_1, n_keep_1, replace=False)
            edges_1_sampled = edges_1[indices_1]
            
            balanced_edges = np.vstack([edges_0, edges_1_sampled])
        else:
            # Class 0 is majority - undersample it
            n_keep_0 = int(n_class_1 * target_ratio / (1 - target_ratio))
            n_keep_0 = min(n_keep_0, n_class_0)  # Don't exceed available samples
            
            indices_0 = np.random.choice(n_class_0, n_keep_0, replace=False)
            edges_0_sampled = edges_0[indices_0]
            
            balanced_edges = np.vstack([edges_0_sampled, edges_1])
        
        # Shuffle
        np.random.shuffle(balanced_edges)
        
        total_balanced += len(balanced_edges)
        
        # Save
        relative_path = npy_file.relative_to(input_path)
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, balanced_edges)
    
    if verbose:
        print(f"\nBalancing complete:")
        print(f"  Original samples: {total_original:,}")
        print(f"  Balanced samples: {total_balanced:,}")
        print(f"  Reduction: {(1 - total_balanced/total_original)*100:.1f}%")


def main():
    # Balance train/val/test splits
    for split in ['train', 'val', 'test']:
        print(f"\n=== Balancing {split} split ===")
        balance_edge_dataset(
            input_dir=f'data/edges/{split}',
            output_dir=f'data/edges_balanced/{split}',
            target_ratio=0.4,
            verbose=True
        )


if __name__ == "__main__":
    main()
