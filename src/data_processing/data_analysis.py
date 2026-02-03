import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def analyze_edges_data(edges_dir: Path, split: str = 'train'):
    """
    Analyze edge data for class balance and feature importance.
    
    Args:
        edges_dir: Path to edges directory
        split: Which split to analyze ('train', 'val', 'test')
    """
    split_dir = edges_dir / split
    npy_files = sorted(split_dir.glob('*.npy'))
    
    if not npy_files:
        print(f"No .npy files found in {split_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"EDGE DATA ANALYSIS - {split.upper()} SPLIT")
    print(f"{'='*60}\n")
    
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
        print("WARNING: Files have different feature dimensions:")
        for shape, files in shapes.items():
            print(f"  {shape} features: {len(files)} files")
        # Use only files with most common shape
        most_common_shape = max(shapes.keys(), key=lambda k: len(shapes[k]))
        print(f"\nUsing only files with {most_common_shape} features ({len(shapes[most_common_shape])} files)")
        all_data = [data for data in all_data if data.shape[1] == most_common_shape]
    
    all_data = np.vstack(all_data)
    X = all_data[:, :-1]  # Features
    y = all_data[:, -1]   # Labels
    
    n_features = X.shape[1]
    n_samples = len(y)
    
    print(f"Total samples: {n_samples:,}")
    print(f"Number of features: {n_features}")
    
    # 1. Class Balance Analysis
    print(f"\n{'='*60}")
    print("CLASS BALANCE ANALYSIS")
    print(f"{'='*60}")
    
    class_0 = np.sum(y == 0)
    class_1 = np.sum(y == 1)
    
    pct_0 = (class_0 / n_samples) * 100
    pct_1 = (class_1 / n_samples) * 100
    
    print(f"\nClass 0 (different trees): {class_0:,} samples ({pct_0:.2f}%)")
    print(f"Class 1 (same tree):       {class_1:,} samples ({pct_1:.2f}%)")
    print(f"Imbalance ratio: {max(class_0, class_1) / min(class_0, class_1):.2f}:1")
    
    # 2. Feature Importance Analysis
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Sample data if too large
    max_samples = 50000
    if n_samples > max_samples:
        print(f"\nSampling {max_samples:,} samples for analysis...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Train Random Forest for feature importance
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_sample, y_sample)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_sample, y_sample, cv=3, scoring='f1')
    
    print(f"\nRandom Forest Performance:")
    print(f"  Cross-val F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Feature names
    n_radii = n_features // 4
    feature_names = []
    for r in range(n_radii):
        feature_names.extend([
            f"R{r+1}_norm_dist",
            f"R{r+1}_angle",
            f"R{r+1}_thickness_ratio",
            f"R{r+1}_vert_diff"
        ])
    
    print(f"\nFeature Importances:")
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {importance:.4f}")
    
    # 3. Feature Separability Analysis
    print(f"\n{'='*60}")
    print("FEATURE SEPARABILITY ANALYSIS")
    print(f"{'='*60}")
    
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    print(f"\nMean feature values by class:")
    print(f"{'Feature':<25} {'Class 0':>12} {'Class 1':>12} {'Difference':>12}")
    print("-" * 65)
    
    for i, name in enumerate(feature_names):
        mean_0 = X_class0[:, i].mean()
        mean_1 = X_class1[:, i].mean()
        diff = abs(mean_0 - mean_1)
        print(f"{name:<25} {mean_0:>12.4f} {mean_1:>12.4f} {diff:>12.4f}")
    
    # Statistical separability
    print(f"\nFeature discriminative power (higher = better):")
    for i, name in enumerate(feature_names):
        std_0 = X_class0[:, i].std()
        std_1 = X_class1[:, i].std()
        mean_0 = X_class0[:, i].mean()
        mean_1 = X_class1[:, i].mean()
        
        # Fisher score: between-class variance / within-class variance
        between_var = (mean_0 - mean_1) ** 2
        within_var = (std_0 ** 2 + std_1 ** 2) / 2
        fisher_score = between_var / (within_var + 1e-10)
        
        print(f"  {name:25s}: {fisher_score:.4f}")
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    
    if cv_scores.mean() > 0.7:
        print("\n✓ Features show GOOD discriminative power (F1 > 0.7)")
        print("  The features can effectively distinguish between classes.")
    elif cv_scores.mean() > 0.5:
        print("\n⚠ Features show MODERATE discriminative power (0.5 < F1 < 0.7)")
        print("  Consider adding more features or feature engineering.")
    else:
        print("\n✗ Features show POOR discriminative power (F1 < 0.5)")
        print("  Features may not be sufficient for classification.")
    
    print()


def main():
    edges_dir = Path("data/edges")
    
    for split in ['train', 'val', 'test']:
        if (edges_dir / split).exists():
            analyze_edges_data(edges_dir, split)


if __name__ == "__main__":
    main()
