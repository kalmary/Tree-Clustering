import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import json


def analyze_edges_data(edges_dir: Path, split: str = 'train', save_stats: bool = True, verbose: bool = True):
    """
    Analyze edge data for class balance, feature importance, and scaling statistics.
    
    Args:
        edges_dir: Path to edges directory
        split: Which split to analyze ('train', 'val', 'test')
        save_stats: If True, save scaling statistics to JSON file
        verbose: If True, print detailed analysis
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
    
    # Load all data and check for consistent dimensions
    all_data = []
    shapes = {}
    for npy_file in npy_files:
        data = np.load(npy_file)
        shape = data.shape[1]
        if shape not in shapes:
            shapes[shape] = []
        shapes[shape].append(npy_file.name)
        all_data.append(data)
    
    # Check if all files have same number of features
    if len(shapes) > 1:
        if verbose:
            print("WARNING: Files have different feature dimensions:")
            for shape, files in shapes.items():
                print(f"  {shape} features: {len(files)} files")
            # Use only files with most common shape
            most_common_shape = max(shapes.keys(), key=lambda k: len(shapes[k]))
            print(f"\nUsing only files with {most_common_shape} features ({len(shapes[most_common_shape])} files)")
        else:
            most_common_shape = max(shapes.keys(), key=lambda k: len(shapes[k]))
        all_data = [data for data in all_data if data.shape[1] == most_common_shape]
    
    all_data = np.vstack(all_data)
    X = all_data[:, :-1]  # Features
    y = all_data[:, -1]   # Labels
    
    n_features = X.shape[1]
    n_samples = len(y)
    
    if verbose:
        print(f"Total samples: {n_samples:,}")
        print(f"Number of features: {n_features}")
    
    # Feature names (customize based on your actual features)
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
    
    # Extend if needed
    while len(feature_names) < n_features:
        feature_names.append(f"feature_{len(feature_names)}")
    
    feature_names = feature_names[:n_features]
    
    # =========================================================================
    # FEATURE SCALING STATISTICS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE SCALING STATISTICS")
        print(f"{'='*80}\n")
    
    # Compute comprehensive statistics
    feature_stats = {}
    
    if verbose:
        print(f"{'Feature':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Median':>10} {'Range':>10}")
        print("-" * 90)
    
    for i, name in enumerate(feature_names):
        feature_col = X[:, i]
        
        stats = {
            'min': float(np.min(feature_col)),
            'max': float(np.max(feature_col)),
            'mean': float(np.mean(feature_col)),
            'std': float(np.std(feature_col)),
            'median': float(np.median(feature_col)),
            'q25': float(np.percentile(feature_col, 25)),
            'q75': float(np.percentile(feature_col, 75)),
            'range': float(np.max(feature_col) - np.min(feature_col))
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
    
    # Analyze which features need scaling
    needs_scaling = []
    bounded_features = []
    
    for i, name in enumerate(feature_names):
        stats = feature_stats[name]
        
        # Check if feature is roughly in [0, 1] range
        if stats['min'] >= -0.1 and stats['max'] <= 1.1:
            bounded_features.append(name)
        else:
            needs_scaling.append(name)
    
    if verbose:
        if bounded_features:
            print(f"✓ Features already in [0, 1] range ({len(bounded_features)}):")
            for name in bounded_features:
                stats = feature_stats[name]
                print(f"  {name:<20} [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        if needs_scaling:
            print(f"\n⚠ Features that need scaling ({len(needs_scaling)}):")
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
        
        # Standard (Z-score) normalization parameters
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
        
        # Min-Max normalization parameters
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
        
        # Robust scaling (using quartiles)
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
            print(f"\n✓ Scaling parameters saved to: {output_file}")
    
    # =========================================================================
    # CLASS BALANCE ANALYSIS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("CLASS BALANCE ANALYSIS")
        print(f"{'='*80}")
    
    class_0 = np.sum(y == 0)
    class_1 = np.sum(y == 1)
    
    pct_0 = (class_0 / n_samples) * 100
    pct_1 = (class_1 / n_samples) * 100
    
    if verbose:
        print(f"\nClass 0 (different trees): {class_0:,} samples ({pct_0:.2f}%)")
        print(f"Class 1 (same tree):       {class_1:,} samples ({pct_1:.2f}%)")
        print(f"Imbalance ratio: {max(class_0, class_1) / min(class_0, class_1):.2f}:1")
    
    # =========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
    
    # Sample data if too large
    max_samples = 50000
    if n_samples > max_samples:
        if verbose:
            print(f"\nSampling {max_samples:,} samples for analysis...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Train Random Forest for feature importance
    if verbose:
        print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_sample, y_sample, cv=3, scoring='f1')
    
    if verbose:
        print(f"\nRandom Forest Performance:")
        print(f"  Cross-val F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print(f"\nFeature Importances:")
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name:25s}: {importance:.4f}")
    
    # =========================================================================
    # FEATURE SEPARABILITY ANALYSIS
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE SEPARABILITY ANALYSIS")
        print(f"{'='*80}")
    
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    if verbose:
        print(f"\nMean feature values by class:")
        print(f"{'Feature':<25} {'Class 0':>12} {'Class 1':>12} {'Difference':>12} {'Effect Size':>12}")
        print("-" * 80)
    
    for i, name in enumerate(feature_names):
        mean_0 = X_class0[:, i].mean()
        mean_1 = X_class1[:, i].mean()
        diff = abs(mean_0 - mean_1)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((X_class0[:, i].std()**2 + X_class1[:, i].std()**2) / 2)
        effect_size = diff / (pooled_std + 1e-10)
        
        if verbose:
            print(f"{name:<25} {mean_0:>12.4f} {mean_1:>12.4f} {diff:>12.4f} {effect_size:>12.4f}")
    
    # Statistical separability (Fisher score)
    if verbose:
        print(f"\nFeature discriminative power (Fisher score - higher = better):")
        print(f"{'Feature':<25} {'Fisher Score':>15} {'Interpretation'}")
        print("-" * 60)
    
    for i, name in enumerate(feature_names):
        std_0 = X_class0[:, i].std()
        std_1 = X_class1[:, i].std()
        mean_0 = X_class0[:, i].mean()
        mean_1 = X_class1[:, i].mean()
        
        # Fisher score: between-class variance / within-class variance
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
    
    return feature_stats


def main():
    edges_dir = Path("data/edges")
    
    for split in ['train']:
        if (edges_dir / split).exists():
            stats = analyze_edges_data(edges_dir, split, save_stats=True, verbose=True)


if __name__ == "__main__":
    main()