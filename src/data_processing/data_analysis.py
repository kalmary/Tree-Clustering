import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
import torch
from tqdm import tqdm
from typing import Iterator, Dict, Tuple, List
from dataclasses import dataclass


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


def load_graph_data_generator(
    pt_files: List[Path],
    batch_size: int = 10000,
    verbose: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generator that yields batches of (edge_features, edge_labels, node_features) from .pt files.
    
    Args:
        pt_files: List of .pt file paths
        batch_size: Target batch size for edge features
        verbose: Show progress bar
        
    Yields:
        Tuple of (edge_features, edge_labels, node_features) arrays
    """
    current_edge_features = []
    current_edge_labels = []
    current_node_features = []
    current_size = 0
    
    pbar = tqdm(pt_files, desc="Loading graph data", disable=not verbose)
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu')
            
            # Extract data from graph
            edge_attr = graph_data['edge_attr'].numpy()
            edge_labels = graph_data['y'].numpy()
            node_features = graph_data['x'].numpy()
            
            # Add to current batch
            current_edge_features.append(edge_attr)
            current_edge_labels.append(edge_labels)
            current_node_features.append(node_features)
            current_size += len(edge_attr)
            
            # Yield when batch is full
            while current_size >= batch_size:
                batch_edge_features = np.vstack(current_edge_features)
                batch_edge_labels = np.concatenate(current_edge_labels)
                batch_node_features = np.vstack(current_node_features)
                
                # Yield exactly batch_size samples
                yield (
                    batch_edge_features[:batch_size],
                    batch_edge_labels[:batch_size],
                    batch_node_features[:min(batch_size, len(batch_node_features))]
                )
                
                # Keep remainder
                if len(batch_edge_features) > batch_size:
                    current_edge_features = [batch_edge_features[batch_size:]]
                    current_edge_labels = [batch_edge_labels[batch_size:]]
                    current_node_features = [batch_node_features]  # Keep all nodes
                    current_size = len(current_edge_features[0])
                else:
                    current_edge_features = []
                    current_edge_labels = []
                    current_node_features = []
                    current_size = 0
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    # Yield remaining data
    if current_edge_features:
        yield (
            np.vstack(current_edge_features),
            np.concatenate(current_edge_labels),
            np.vstack(current_node_features) if current_node_features else np.array([])
        )


def compute_quantiles_streaming(
    pt_files: List[Path],
    percentiles: List[int] = [25, 50, 75],
    n_features: int = 8,
    sample_size: int = 100000,
    verbose: bool = True
) -> Dict[int, np.ndarray]:
    """
    Compute quantiles by sampling from edge features (approximation for large datasets).
    """
    # Reservoir sampling to get representative sample
    samples = []
    total_seen = 0
    
    pbar = tqdm(pt_files, desc="Sampling for quantiles", disable=not verbose)
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu')
            edge_features = graph_data['edge_attr'].numpy()
            
            for i in range(len(edge_features)):
                total_seen += 1
                
                if len(samples) < sample_size:
                    samples.append(edge_features[i])
                else:
                    # Reservoir sampling: randomly replace
                    j = np.random.randint(0, total_seen)
                    if j < sample_size:
                        samples[j] = edge_features[i]
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    if not samples:
        return {p: np.zeros(n_features) for p in percentiles}
    
    # Compute quantiles from sample
    samples_array = np.array(samples)
    quantiles = {}
    
    for p in percentiles:
        quantiles[p] = np.percentile(samples_array, p, axis=0)
    
    return quantiles


def compute_class_statistics_streaming(
    pt_files: List[Path],
    n_features: int = 8,
    verbose: bool = True
) -> Tuple[OnlineStats, OnlineStats, int, int]:
    """
    Compute per-class statistics for edge features using streaming approach.
    
    Returns:
        (class_0_stats, class_1_stats, class_0_count, class_1_count)
    """
    class_0_stats = OnlineStats()
    class_1_stats = OnlineStats()
    class_0_count = 0
    class_1_count = 0
    
    pbar = tqdm(pt_files, desc="Computing class statistics", disable=not verbose)
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu')
            edge_features = graph_data['edge_attr'].numpy()
            edge_labels = graph_data['y'].numpy()
            
            # Split by class
            class_0_mask = edge_labels == 0
            class_1_mask = edge_labels == 1
            
            class_0_features = edge_features[class_0_mask]
            class_1_features = edge_features[class_1_mask]
            
            class_0_count += len(class_0_features)
            class_1_count += len(class_1_features)
            
            if len(class_0_features) > 0:
                class_0_stats.update(class_0_features)
            if len(class_1_features) > 0:
                class_1_stats.update(class_1_features)
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    return class_0_stats, class_1_stats, class_0_count, class_1_count


def analyze_graph_data(
    edges_dir: Path,
    split: str = 'train',
    save_stats: bool = True,
    verbose: bool = True,
    max_samples_rf: int = 50000
):
    """
    Memory-efficient analysis of graph edge data from .pt files.
    
    Args:
        edges_dir: Path to edges directory containing .pt files
        split: Which split to analyze ('train', 'val', 'test')
        save_stats: If True, save scaling statistics to JSON file
        verbose: If True, print detailed analysis
        max_samples_rf: Maximum samples for Random Forest training
    """
    split_dir = edges_dir / split
    pt_files = sorted(split_dir.glob('*.pt'))
    
    if not pt_files:
        if verbose:
            print(f"No .pt files found in {split_dir}")
        return
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GRAPH EDGE DATA ANALYSIS - {split.upper()} SPLIT")
        print(f"{'='*80}\n")
        print(f"Found {len(pt_files)} graph files")
        print(f"Memory-efficient mode: Processing in batches")
    
    # Check dimensions and total sample count
    if verbose:
        print(f"\nScanning files for dimensions...")
    
    total_edges = 0
    total_nodes = 0
    n_features = None
    n_node_features = None
    
    pbar = tqdm(pt_files, desc="Scanning files", disable=not verbose) if verbose else pt_files
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu')
            total_edges += graph_data['edge_attr'].shape[0]
            total_nodes += graph_data['num_nodes']
            
            if n_features is None:
                n_features = graph_data['edge_attr'].shape[1]
                n_node_features = graph_data['x'].shape[1]
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    if verbose:
        print(f"\nTotal edges: {total_edges:,}")
        print(f"Total nodes: {total_nodes:,}")
        print(f"Number of edge features: {n_features}")
        print(f"Number of node features: {n_node_features}")
    
    # Feature names for edge features
    edge_feature_names = [
        "distance",
        "angle",
        "thickness_ratio",
        "vertical_diff",
        "vertical_offset",
        "density_ratio",
        "height_ratio",
        "mean_height"
    ]
    
    # Adjust if different number of features
    while len(edge_feature_names) < n_features:
        edge_feature_names.append(f"feature_{len(edge_feature_names)}")
    
    edge_feature_names = edge_feature_names[:n_features]
    
    # Node feature names
    node_feature_names = [
        "thickness",
        "verticality",
        "linearity",
        "planarity",
        "scattering",
        "height",
        "point_count"
    ]
    
    while len(node_feature_names) < n_node_features:
        node_feature_names.append(f"node_feature_{len(node_feature_names)}")
    
    node_feature_names = node_feature_names[:n_node_features]
    
    # =========================================================================
    # STREAMING STATISTICS COMPUTATION - EDGE FEATURES
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING EDGE FEATURE STATISTICS (streaming mode)")
        print(f"{'='*80}\n")
    
    # Compute online statistics
    edge_stats = OnlineStats()
    
    for edge_features, _, _ in load_graph_data_generator(pt_files, batch_size=10000, verbose=verbose):
        edge_stats.update(edge_features)
    
    # Compute quantiles via sampling
    if verbose:
        print("\nComputing quantiles (using reservoir sampling)...")
    
    quantiles = compute_quantiles_streaming(
        pt_files,
        percentiles=[25, 50, 75],
        n_features=n_features,
        sample_size=min(100000, total_edges),
        verbose=verbose
    )
    
    # Build feature statistics dictionary
    feature_stats = {}
    
    if verbose:
        print(f"\n{'='*80}")
        print("EDGE FEATURE SCALING STATISTICS")
        print(f"{'='*80}\n")
        print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Median':>10} {'Range':>10}")
        print("-" * 90)
    
    for i, name in enumerate(edge_feature_names):
        stats = {
            'min': float(edge_stats.min_val[i]),
            'max': float(edge_stats.max_val[i]),
            'mean': float(edge_stats.mean[i]),
            'std': float(edge_stats.std[i]),
            'median': float(quantiles[50][i]),
            'q25': float(quantiles[25][i]),
            'q75': float(quantiles[75][i]),
            'range': float(edge_stats.max_val[i] - edge_stats.min_val[i])
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
    
    for i, name in enumerate(edge_feature_names):
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
        print("1. STANDARD SCALING (recommended for GNNs):")
        print("   Formula: (x - mean) / std")
        print("\n   Python code:")
        print("   ```python")
        print("   means = np.array([")
        means_str = ", ".join([f"{feature_stats[name]['mean']:.6f}" for name in edge_feature_names])
        print(f"       {means_str}")
        print("   ])")
        print("   stds = np.array([")
        stds_str = ", ".join([f"{feature_stats[name]['std']:.6f}" for name in edge_feature_names])
        print(f"       {stds_str}")
        print("   ])")
        print("   edge_features_normalized = (edge_features - means) / (stds + 1e-8)")
        print("   ```")
        
        # Min-Max scaling
        print("\n2. MIN-MAX SCALING (0-1 range):")
        print("   Formula: (x - min) / (max - min)")
        print("\n   Python code:")
        print("   ```python")
        print("   mins = np.array([")
        mins_str = ", ".join([f"{feature_stats[name]['min']:.6f}" for name in edge_feature_names])
        print(f"       {mins_str}")
        print("   ])")
        print("   maxs = np.array([")
        maxs_str = ", ".join([f"{feature_stats[name]['max']:.6f}" for name in edge_feature_names])
        print(f"       {maxs_str}")
        print("   ])")
        print("   edge_features_normalized = (edge_features - mins) / (maxs - mins + 1e-8)")
        print("   ```")
    
    # Save to JSON
    if save_stats:
        scaling_params = {
            'feature_names': edge_feature_names,
            'node_feature_names': node_feature_names,
            'n_edge_features': n_features,
            'n_node_features': n_node_features,
            'n_edges': total_edges,
            'n_nodes': total_nodes,
            'standard_scaling': {
                'means': [feature_stats[name]['mean'] for name in edge_feature_names],
                'stds': [feature_stats[name]['std'] for name in edge_feature_names]
            },
            'minmax_scaling': {
                'mins': [feature_stats[name]['min'] for name in edge_feature_names],
                'maxs': [feature_stats[name]['max'] for name in edge_feature_names]
            },
            'robust_scaling': {
                'medians': [feature_stats[name]['median'] for name in edge_feature_names],
                'q25s': [feature_stats[name]['q25'] for name in edge_feature_names],
                'q75s': [feature_stats[name]['q75'] for name in edge_feature_names]
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
    
    for pt_file in tqdm(pt_files, desc="Counting classes", disable=not verbose):
        try:
            graph_data = torch.load(pt_file, map_location='cpu')
            labels = graph_data['y'].numpy()
            class_0_count += int(np.sum(labels == 0))
            class_1_count += int(np.sum(labels == 1))
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    total_samples = class_0_count + class_1_count
    pct_0 = (class_0_count / total_samples) * 100 if total_samples > 0 else 0
    pct_1 = (class_1_count / total_samples) * 100 if total_samples > 0 else 0
    
    if verbose:
        print(f"Class 0 (different trees): {class_0_count:,} samples ({pct_0:.2f}%)")
        print(f"Class 1 (same tree):       {class_1_count:,} samples ({pct_1:.2f}%)")
        if min(class_0_count, class_1_count) > 0:
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
    
    for pt_file in tqdm(pt_files, desc="Collecting RF sample", disable=not verbose):
        if samples_collected >= max_samples_rf:
            break
        
        try:
            graph_data = torch.load(pt_file, map_location='cpu')
            edge_features = graph_data['edge_attr'].numpy()
            edge_labels = graph_data['y'].numpy()
            
            # Sample from this file
            n_to_sample = min(len(edge_features), max_samples_rf - samples_collected)
            
            if n_to_sample < len(edge_features):
                indices = np.random.choice(len(edge_features), n_to_sample, replace=False)
                sample_features.append(edge_features[indices])
                sample_labels.append(edge_labels[indices])
            else:
                sample_features.append(edge_features)
                sample_labels.append(edge_labels)
            
            samples_collected += n_to_sample
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    if sample_features:
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
            for name, importance in sorted(zip(edge_feature_names, importances), key=lambda x: x[1], reverse=True):
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
        pt_files, n_features, verbose
    )
    
    if verbose:
        print(f"\nMean feature values by class:")
        print(f"{'Feature':<25} {'Class 0':>12} {'Class 1':>12} {'Difference':>12} {'Effect Size':>12}")
        print("-" * 80)
    
    for i, name in enumerate(edge_feature_names):
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
    
    for i, name in enumerate(edge_feature_names):
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
    
    # =========================================================================
    # GRAPH-SPECIFIC METRICS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("GRAPH STRUCTURE ANALYSIS")
        print(f"{'='*80}\n")
        
        avg_edges_per_graph = total_edges / len(pt_files)
        avg_nodes_per_graph = total_nodes / len(pt_files)
        avg_edges_per_node = total_edges / total_nodes if total_nodes > 0 else 0
        
        print(f"Average edges per graph: {avg_edges_per_graph:.1f}")
        print(f"Average nodes per graph: {avg_nodes_per_graph:.1f}")
        print(f"Average degree (edges/node): {avg_edges_per_node:.2f}")
    
    return feature_stats, scaling_params if save_stats else None


def main():
    edges_dir = Path("data/edges")
    
    for split in ['train']:
        if (edges_dir / split).exists():
            print(f"\n{'='*80}")
            print(f"Processing {split} split...")
            print(f"{'='*80}")
            
            stats, scaling_params = analyze_graph_data(
                edges_dir,
                split,
                save_stats=True,
                verbose=True,
                max_samples_rf=50000
            )


if __name__ == "__main__":
    main()