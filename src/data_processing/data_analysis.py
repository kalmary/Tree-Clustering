import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
from tqdm import tqdm
from typing import Iterator, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class OnlineStats:
    """Compute statistics incrementally without storing all data."""
    count: int = 0
    mean: np.ndarray = None
    m2: np.ndarray = None  # For variance calculation
    min_val: np.ndarray = None
    max_val: np.ndarray = None
    
    def update(self, batch: np.ndarray):
        """Update statistics with a new batch of data using Welford's online algorithm."""
        batch_size = len(batch)
        
        if self.mean is None:
            # Initialize
            n_features = batch.shape[1]
            self.mean = np.zeros(n_features)
            self.m2 = np.zeros(n_features)
            self.min_val = np.full(n_features, np.inf)
            self.max_val = np.full(n_features, -np.inf)
        
        # Update min/max
        self.min_val = np.minimum(self.min_val, batch.min(axis=0))
        self.max_val = np.maximum(self.max_val, batch.max(axis=0))
        
        # Welford's online algorithm for mean and variance
        for x in batch:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
    
    @property
    def variance(self):
        """Get variance."""
        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.m2 / self.count
    
    @property
    def std(self):
        """Get standard deviation."""
        return np.sqrt(self.variance)


def load_data_generator(
    npy_files: list,
    batch_size: int = 10000,
    verbose: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator that yields batches of (features, labels) from files.
    
    Args:
        npy_files: List of .npy file paths
        batch_size: Target batch size (accumulated across files)
        verbose: Show progress bar
        
    Yields:
        Tuple of (features, labels) arrays
    """
    current_features = []
    current_labels = []
    current_size = 0
    
    pbar = tqdm(npy_files, desc="Loading data", disable=not verbose)
    
    for npy_file in pbar:
        data = np.load(npy_file)
        
        # Split features and labels
        features = data[:, :-1]
        labels = data[:, -1]
        
        # Add to current batch
        current_features.append(features)
        current_labels.append(labels)
        current_size += len(features)
        
        # Yield when batch is full
        while current_size >= batch_size:
            batch_features = np.vstack(current_features)
            batch_labels = np.concatenate(current_labels)
            
            # Yield exactly batch_size samples
            yield batch_features[:batch_size], batch_labels[:batch_size]
            
            # Keep remainder
            if len(batch_features) > batch_size:
                current_features = [batch_features[batch_size:]]
                current_labels = [batch_labels[batch_size:]]
                current_size = len(current_features[0])
            else:
                current_features = []
                current_labels = []
                current_size = 0
    
    # Yield remaining data
    if current_features:
        yield np.vstack(current_features), np.concatenate(current_labels)


def compute_quantiles_streaming(
    npy_files: list,
    percentiles: list = [25, 50, 75],
    n_features: int = 8,
    sample_size: int = 100000,
    verbose: bool = True
) -> Dict[int, np.ndarray]:
    """
    Compute quantiles by sampling from data (approximation for large datasets).
    
    Args:
        npy_files: List of .npy file paths
        percentiles: List of percentiles to compute (e.g., [25, 50, 75])
        n_features: Number of features
        sample_size: Number of samples to use for quantile estimation
        verbose: Show progress
        
    Returns:
        Dictionary mapping percentile to feature quantile values
    """
    # Reservoir sampling to get representative sample
    samples = []
    total_seen = 0
    
    pbar = tqdm(npy_files, desc="Sampling for quantiles", disable=not verbose)
    
    for npy_file in pbar:
        data = np.load(npy_file, mmap_mode='r')
        features = data[:, :-1]
        
        for i in range(len(features)):
            total_seen += 1
            
            if len(samples) < sample_size:
                samples.append(features[i])
            else:
                # Reservoir sampling: randomly replace
                j = np.random.randint(0, total_seen)
                if j < sample_size:
                    samples[j] = features[i]
    
    if not samples:
        return {p: np.zeros(n_features) for p in percentiles}
    
    # Compute quantiles from sample
    samples_array = np.array(samples)
    quantiles = {}
    
    for p in percentiles:
        quantiles[p] = np.percentile(samples_array, p, axis=0)
    
    return quantiles


def compute_class_statistics_streaming(
    npy_files: list,
    n_features: int = 8,
    verbose: bool = True
) -> Tuple[OnlineStats, OnlineStats, int, int]:
    """
    Compute per-class statistics using streaming approach.
    
    Returns:
        (class_0_stats, class_1_stats, class_0_count, class_1_count)
    """
    class_0_stats = OnlineStats()
    class_1_stats = OnlineStats()
    class_0_count = 0
    class_1_count = 0
    
    pbar = tqdm(npy_files, desc="Computing class statistics", disable=not verbose)
    
    for npy_file in pbar:
        data = np.load(npy_file)
        features = data[:, :-1]
        labels = data[:, -1]
        
        # Split by class
        class_0_mask = labels == 0
        class_1_mask = labels == 1
        
        class_0_features = features[class_0_mask]
        class_1_features = features[class_1_mask]
        
        class_0_count += len(class_0_features)
        class_1_count += len(class_1_features)
        
        if len(class_0_features) > 0:
            class_0_stats.update(class_0_features)
        if len(class_1_features) > 0:
            class_1_stats.update(class_1_features)
    
    return class_0_stats, class_1_stats, class_0_count, class_1_count


def analyze_edges_data(
    edges_dir: Path,
    split: str = 'train',
    save_stats: bool = True,
    verbose: bool = True,
    max_samples_rf: int = 50000,
    max_memory_gb: float = 40.0
):
    """
    Memory-efficient analysis of edge data.
    
    Args:
        edges_dir: Path to edges directory
        split: Which split to analyze ('train', 'val', 'test')
        save_stats: If True, save scaling statistics to JSON file
        verbose: If True, print detailed analysis
        max_samples_rf: Maximum samples for Random Forest training
        max_memory_gb: Target maximum memory usage
    """
    split_dir = edges_dir / split
    npy_files = sorted(split_dir.glob('*.npy'))
    
    if not npy_files:
        if verbose:
            print(f"No .npy files found in {split_dir}")
        return
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"EDGE DATA ANALYSIS - {split.upper()} SPLIT")
        print(f"{'='*80}\n")
        print(f"Found {len(npy_files)} files")
        print(f"Memory-efficient mode: Processing in batches")
    
    # Check dimensions and total sample count
    if verbose:
        print(f"\nScanning files for dimensions...")
    
    shapes = {}
    total_samples = 0
    
    pbar = tqdm(npy_files, desc="Scanning files", disable=not verbose) if verbose else npy_files
    
    for npy_file in pbar:
        data = np.load(npy_file, mmap_mode='r')
        shape = data.shape[1]
        total_samples += len(data)
        
        if shape not in shapes:
            shapes[shape] = []
        shapes[shape].append(npy_file.name)
    
    # Check dimension consistency
    if len(shapes) > 1:
        if verbose:
            print("WARNING: Files have different feature dimensions:")
            for shape, files in shapes.items():
                print(f"  {shape} features: {len(files)} files")
        most_common_shape = max(shapes.keys(), key=lambda k: len(shapes[k]))
        if verbose:
            print(f"\nUsing only files with {most_common_shape} features")
        npy_files = [edges_dir / split / f for f in shapes[most_common_shape]]
    else:
        most_common_shape = list(shapes.keys())[0]
    
    n_features = most_common_shape - 1  # Exclude label column
    
    if verbose:
        print(f"\nTotal samples: {total_samples:,}")
        print(f"Number of features: {n_features}")
    
    # Feature names
    feature_names = [
        "distance",
        "angle",
        "thickness_ratio",
        "vertical_diff",
        "vertical_offset",
        "density_ratio",
        "height_ratio",
        "mean_height"
    ]
    
    while len(feature_names) < n_features:
        feature_names.append(f"feature_{len(feature_names)}")
    
    feature_names = feature_names[:n_features]
    
    # =========================================================================
    # STREAMING STATISTICS COMPUTATION
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING FEATURE STATISTICS (streaming mode)")
        print(f"{'='*80}\n")
    
    # Compute online statistics
    overall_stats = OnlineStats()
    
    for features, labels in load_data_generator(npy_files, batch_size=10000, verbose=verbose):
        overall_stats.update(features)
    
    # Compute quantiles via sampling
    if verbose:
        print("\nComputing quantiles (using reservoir sampling)...")
    
    quantiles = compute_quantiles_streaming(
        npy_files,
        percentiles=[25, 50, 75],
        n_features=n_features,
        sample_size=min(100000, total_samples),
        verbose=verbose
    )
    
    # Build feature statistics dictionary
    feature_stats = {}
    
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE SCALING STATISTICS")
        print(f"{'='*80}\n")
        print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Median':>10} {'Range':>10}")
        print("-" * 90)
    
    for i, name in enumerate(feature_names):
        stats = {
            'min': float(overall_stats.min_val[i]),
            'max': float(overall_stats.max_val[i]),
            'mean': float(overall_stats.mean[i]),
            'std': float(overall_stats.std[i]),
            'median': float(quantiles[50][i]),
            'q25': float(quantiles[25][i]),
            'q75': float(quantiles[75][i]),
            'range': float(overall_stats.max_val[i] - overall_stats.min_val[i])
        }
        
        feature_stats[name] = stats
        
        if verbose:
            print(f"{name:<20} {stats['min']:>10.4f} {stats['max']:>10.4f} {stats['mean']:>10.4f} "
                  f"{stats['std']:>10.4f} {stats['median']:>10.4f} {stats['range']:>10.4f}")
    
    # =========================================================================
    # SCALING RECOMMENDATIONS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("SCALING RECOMMENDATIONS")
        print(f"{'='*80}\n")
    
    needs_scaling = []
    bounded_features = []
    
    for i, name in enumerate(feature_names):
        stats = feature_stats[name]
        
        if stats['min'] >= -0.1 and stats['max'] <= 1.1:
            bounded_features.append(name)
        else:
            needs_scaling.append(name)
    
    if verbose:
        if bounded_features:
            print(f"Features already in [0, 1] range ({len(bounded_features)}):")
            for name in bounded_features:
                stats = feature_stats[name]
                print(f"  {name:<20} [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        if needs_scaling:
            print(f"\nFeatures that need scaling ({len(needs_scaling)}):")
            for name in needs_scaling:
                stats = feature_stats[name]
                print(f"  {name:<20} [{stats['min']:.4f}, {stats['max']:.4f}] - range: {stats['range']:.4f}")
    
    # =========================================================================
    # NORMALIZATION PARAMETERS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("NORMALIZATION PARAMETERS FOR YOUR MODEL")
        print(f"{'='*80}\n")
        
        # Standard scaling
        print("1. STANDARD SCALING (recommended):")
        print("   Formula: (x - mean) / std")
        print("\n   Python code:")
        print("   ```python")
        print("   means = np.array([")
        means_str = ", ".join([f"{feature_stats[name]['mean']:.6f}" for name in feature_names])
        print(f"       {means_str}")
        print("   ])")
        print("   stds = np.array([")
        stds_str = ", ".join([f"{feature_stats[name]['std']:.6f}" for name in feature_names])
        print(f"       {stds_str}")
        print("   ])")
        print("   features_normalized = (features - means) / (stds + 1e-8)")
        print("   ```")
        
        # Min-Max scaling
        print("\n2. MIN-MAX SCALING (0-1 range):")
        print("   Formula: (x - min) / (max - min)")
        print("\n   Python code:")
        print("   ```python")
        print("   mins = np.array([")
        mins_str = ", ".join([f"{feature_stats[name]['min']:.6f}" for name in feature_names])
        print(f"       {mins_str}")
        print("   ])")
        print("   maxs = np.array([")
        maxs_str = ", ".join([f"{feature_stats[name]['max']:.6f}" for name in feature_names])
        print(f"       {maxs_str}")
        print("   ])")
        print("   features_normalized = (features - mins) / (maxs - mins + 1e-8)")
        print("   ```")
        
        # Robust scaling
        print("\n3. ROBUST SCALING (outlier-resistant):")
        print("   Formula: (x - median) / (Q75 - Q25)")
        print("\n   Python code:")
        print("   ```python")
        print("   medians = np.array([")
        medians_str = ", ".join([f"{feature_stats[name]['median']:.6f}" for name in feature_names])
        print(f"       {medians_str}")
        print("   ])")
        print("   q25s = np.array([")
        q25s_str = ", ".join([f"{feature_stats[name]['q25']:.6f}" for name in feature_names])
        print(f"       {q25s_str}")
        print("   ])")
        print("   q75s = np.array([")
        q75s_str = ", ".join([f"{feature_stats[name]['q75']:.6f}" for name in feature_names])
        print(f"       {q75s_str}")
        print("   ])")
        print("   iqr = q75s - q25s")
        print("   features_normalized = (features - medians) / (iqr + 1e-8)")
        print("   ```")
    
    # Save to JSON
    if save_stats:
        scaling_params = {
            'feature_names': feature_names,
            'n_features': n_features,
            'n_samples': total_samples,
            'standard_scaling': {
                'means': [feature_stats[name]['mean'] for name in feature_names],
                'stds': [feature_stats[name]['std'] for name in feature_names]
            },
            'minmax_scaling': {
                'mins': [feature_stats[name]['min'] for name in feature_names],
                'maxs': [feature_stats[name]['max'] for name in feature_names]
            },
            'robust_scaling': {
                'medians': [feature_stats[name]['median'] for name in feature_names],
                'q25s': [feature_stats[name]['q25'] for name in feature_names],
                'q75s': [feature_stats[name]['q75'] for name in feature_names]
            },
            'feature_statistics': feature_stats
        }
        
        output_file = edges_dir / f'scaling_params_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(scaling_params, f, indent=2)
        
        if verbose:
            print(f"\nScaling parameters saved to: {output_file}")
    
    # =========================================================================
    # CLASS BALANCE ANALYSIS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("CLASS BALANCE ANALYSIS")
        print(f"{'='*80}\n")
    
    # Count classes via streaming
    class_0_count = 0
    class_1_count = 0
    
    for npy_file in tqdm(npy_files, desc="Counting classes", disable=not verbose):
        data = np.load(npy_file, mmap_mode='r')
        labels = data[:, -1]
        class_0_count += int(np.sum(labels == 0))
        class_1_count += int(np.sum(labels == 1))
    
    pct_0 = (class_0_count / total_samples) * 100
    pct_1 = (class_1_count / total_samples) * 100
    
    if verbose:
        print(f"Class 0 (different trees): {class_0_count:,} samples ({pct_0:.2f}%)")
        print(f"Class 1 (same tree):       {class_1_count:,} samples ({pct_1:.2f}%)")
        print(f"Imbalance ratio: {max(class_0_count, class_1_count) / min(class_0_count, class_1_count):.2f}:1")
    
    # =========================================================================
    # FEATURE IMPORTANCE ANALYSIS (with sampling)
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Sampling {max_samples_rf:,} samples for Random Forest...")
    
    # Collect sample for RF training
    sample_features = []
    sample_labels = []
    samples_collected = 0
    
    for npy_file in tqdm(npy_files, desc="Collecting RF sample", disable=not verbose):
        if samples_collected >= max_samples_rf:
            break
        
        data = np.load(npy_file)
        features = data[:, :-1]
        labels = data[:, -1]
        
        # Sample from this file
        n_to_sample = min(len(data), max_samples_rf - samples_collected)
        
        if n_to_sample < len(data):
            indices = np.random.choice(len(data), n_to_sample, replace=False)
            sample_features.append(features[indices])
            sample_labels.append(labels[indices])
        else:
            sample_features.append(features)
            sample_labels.append(labels)
        
        samples_collected += n_to_sample
    
    X_sample = np.vstack(sample_features)
    y_sample = np.concatenate(sample_labels)
    
    if verbose:
        print(f"Training Random Forest on {len(X_sample):,} samples...")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    
    importances = rf.feature_importances_
    
    cv_scores = cross_val_score(rf, X_sample, y_sample, cv=3, scoring='f1')
    
    if verbose:
        print(f"\nRandom Forest Performance:")
        print(f"  Cross-val F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print(f"\nFeature Importances:")
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name:25s}: {importance:.4f}")
    
    # =========================================================================
    # FEATURE SEPARABILITY ANALYSIS (streaming)
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE SEPARABILITY ANALYSIS")
        print(f"{'='*80}\n")
        print("Computing per-class statistics...")
    
    class_0_stats, class_1_stats, c0_count, c1_count = compute_class_statistics_streaming(
        npy_files, n_features, verbose
    )
    
    if verbose:
        print(f"\nMean feature values by class:")
        print(f"{'Feature':<25} {'Class 0':>12} {'Class 1':>12} {'Difference':>12} {'Effect Size':>12}")
        print("-" * 80)
    
    for i, name in enumerate(feature_names):
        mean_0 = class_0_stats.mean[i]
        mean_1 = class_1_stats.mean[i]
        diff = abs(mean_0 - mean_1)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((class_0_stats.variance[i] + class_1_stats.variance[i]) / 2)
        effect_size = diff / (pooled_std + 1e-10)
        
        if verbose:
            print(f"{name:<25} {mean_0:>12.4f} {mean_1:>12.4f} {diff:>12.4f} {effect_size:>12.4f}")
    
    # Fisher score
    if verbose:
        print(f"\nFeature discriminative power (Fisher score - higher = better):")
        print(f"{'Feature':<25} {'Fisher Score':>15} {'Interpretation'}")
        print("-" * 60)
    
    for i, name in enumerate(feature_names):
        std_0 = class_0_stats.std[i]
        std_1 = class_1_stats.std[i]
        mean_0 = class_0_stats.mean[i]
        mean_1 = class_1_stats.mean[i]
        
        between_var = (mean_0 - mean_1) ** 2
        within_var = (std_0 ** 2 + std_1 ** 2) / 2
        fisher_score = between_var / (within_var + 1e-10)
        
        if verbose:
            if fisher_score > 1.0:
                interpretation = "Excellent"
            elif fisher_score > 0.5:
                interpretation = "Good"
            elif fisher_score > 0.1:
                interpretation = "Moderate"
            else:
                interpretation = "Poor"
            
            print(f"  {name:<25} {fisher_score:>15.4f} {interpretation}")
    
    return feature_stats, scaling_params if save_stats else None


def main():
    edges_dir = Path("data/edges")
    
    for split in ['train']:
        if (edges_dir / split).exists():
            print(f"\n{'='*80}")
            print(f"Processing {split} split...")
            print(f"{'='*80}")
            
            stats, scaling_params = analyze_edges_data(
                edges_dir,
                split,
                save_stats=True,
                verbose=True,
                max_samples_rf=50000  # Limit RF training samples
            )


if __name__ == "__main__":
    main()