import numpy as np
import sys
import pathlib as pth
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import json
import torch
from tqdm import tqdm
from typing import Iterator, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict


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
            n_features = batch.shape[1]
            self.mean = np.zeros(n_features)
            self.m2 = np.zeros(n_features)
            self.min_val = np.full(n_features, np.inf)
            self.max_val = np.full(n_features, -np.inf)
        
        self.min_val = np.minimum(self.min_val, batch.min(axis=0))
        self.max_val = np.maximum(self.max_val, batch.max(axis=0))
        
        for x in batch:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
    
    @property
    def variance(self):
        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.m2 / self.count
    
    @property
    def std(self):
        return np.sqrt(self.variance)


def load_graph_data_generator(
    pt_files: List[Path],
    batch_size: int = 10000,
    verbose: bool = True
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generator that yields batches of (edge_features, edge_labels, node_features) from .pt files.
    """
    current_edge_features = []
    current_edge_labels = []
    current_node_features = []
    current_size = 0
    
    pbar = tqdm(pt_files, desc="Loading graph data", disable=not verbose)
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            
            edge_attr = graph_data['edge_attr'].numpy()
            edge_labels = graph_data['y'].numpy()
            node_features = graph_data['x'].numpy()
            
            current_edge_features.append(edge_attr)
            current_edge_labels.append(edge_labels)
            current_node_features.append(node_features)
            current_size += len(edge_attr)
            
            while current_size >= batch_size:
                batch_edge_features = np.vstack(current_edge_features)
                batch_edge_labels = np.concatenate(current_edge_labels)
                batch_node_features = np.vstack(current_node_features)
                
                yield (
                    batch_edge_features[:batch_size],
                    batch_edge_labels[:batch_size],
                    batch_node_features
                )
                
                if len(batch_edge_features) > batch_size:
                    current_edge_features = [batch_edge_features[batch_size:]]
                    current_edge_labels = [batch_edge_labels[batch_size:]]
                    current_node_features = [batch_node_features]
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
    
    if current_edge_features:
        yield (
            np.vstack(current_edge_features),
            np.concatenate(current_edge_labels),
            np.vstack(current_node_features) if current_node_features else np.array([])
        )


def compute_quantiles_streaming(
    pt_files: List[Path],
    percentiles: List[int] = [25, 50, 75],
    n_edge_features: int = 12,
    n_node_features: int = 10,
    sample_size: int = 100000,
    verbose: bool = True
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute quantiles for both edge and node features via reservoir sampling.

    Returns:
        (edge_quantiles, node_quantiles) — both dicts mapping percentile -> array
    """
    edge_samples = []
    node_samples = []
    total_edges_seen = 0
    total_nodes_seen = 0

    pbar = tqdm(pt_files, desc="Sampling for quantiles", disable=not verbose)

    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            edge_features = graph_data['edge_attr'].numpy()
            node_features = graph_data['x'].numpy()

            # Reservoir sampling for edges
            for feat in edge_features:
                total_edges_seen += 1
                if len(edge_samples) < sample_size:
                    edge_samples.append(feat)
                else:
                    j = np.random.randint(0, total_edges_seen)
                    if j < sample_size:
                        edge_samples[j] = feat

            # Reservoir sampling for nodes
            for feat in node_features:
                total_nodes_seen += 1
                if len(node_samples) < sample_size:
                    node_samples.append(feat)
                else:
                    j = np.random.randint(0, total_nodes_seen)
                    if j < sample_size:
                        node_samples[j] = feat

        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue

    edge_quantiles = {}
    node_quantiles = {}

    if edge_samples:
        arr = np.array(edge_samples)
        for p in percentiles:
            edge_quantiles[p] = np.percentile(arr, p, axis=0)
    else:
        for p in percentiles:
            edge_quantiles[p] = np.zeros(n_edge_features)

    if node_samples:
        arr = np.array(node_samples)
        for p in percentiles:
            node_quantiles[p] = np.percentile(arr, p, axis=0)
    else:
        for p in percentiles:
            node_quantiles[p] = np.zeros(n_node_features)

    return edge_quantiles, node_quantiles


def compute_class_statistics_streaming(
    pt_files: List[Path],
    n_features: int = 12,
    verbose: bool = True
) -> Tuple[OnlineStats, OnlineStats, int, int]:
    """Compute per-class statistics for edge features using streaming approach."""
    class_0_stats = OnlineStats()
    class_1_stats = OnlineStats()
    class_0_count = 0
    class_1_count = 0
    
    pbar = tqdm(pt_files, desc="Computing class statistics", disable=not verbose)
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            edge_features = graph_data['edge_attr'].numpy()
            edge_labels = graph_data['y'].numpy()
            
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
    Memory-efficient analysis of graph data from .pt files.
    Computes separate scaling statistics for node features (10) and edge features (12).
    """
    split_dir = edges_dir / split
    pt_files = sorted(split_dir.glob('*.pt'))
    
    if not pt_files:
        if verbose:
            print(f"No .pt files found in {split_dir}")
        return
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"GRAPH DATA ANALYSIS - {split.upper()} SPLIT")
        print(f"{'='*80}\n")
        print(f"Found {len(pt_files)} graph files")
    
    # =========================================================================
    # SCAN FILES FOR BASIC STATISTICS
    # =========================================================================
    total_edges = 0
    total_nodes = 0
    n_edge_features = None
    n_node_features = None
    edges_per_graph = []
    nodes_per_graph = []
    single_class_graphs = 0
    mixed_class_graphs = 0
    class_0_only = 0
    class_1_only = 0
    
    pbar = tqdm(pt_files, desc="Scanning files", disable=not verbose)
    
    for pt_file in pbar:
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            n_edges = graph_data['edge_attr'].shape[0]
            n_nodes = graph_data['num_nodes']
            
            total_edges += n_edges
            total_nodes += n_nodes
            edges_per_graph.append(n_edges)
            nodes_per_graph.append(n_nodes)
            
            labels = graph_data['y'].numpy()
            class_0 = (labels == 0).sum()
            class_1 = (labels == 1).sum()
            
            if class_0 == 0 or class_1 == 0:
                single_class_graphs += 1
                if class_0 == 0:
                    class_1_only += 1
                else:
                    class_0_only += 1
            else:
                mixed_class_graphs += 1
            
            if n_edge_features is None:
                n_edge_features = graph_data['edge_attr'].shape[1]
                n_node_features = graph_data['x'].shape[1]
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue
    
    if verbose:
        print(f"\nDataset Overview:")
        print(f"  Total graphs:        {len(pt_files):,}")
        print(f"  Total edges:         {total_edges:,}")
        print(f"  Total nodes:         {total_nodes:,}")
        print(f"  Edge features:       {n_edge_features}")
        print(f"  Node features:       {n_node_features}")
        print(f"  Mixed-class graphs:  {mixed_class_graphs:,} ({mixed_class_graphs/len(pt_files)*100:.1f}%)")
        print(f"  Single-class graphs: {single_class_graphs:,} ({single_class_graphs/len(pt_files)*100:.1f}%)")

    # =========================================================================
    # FEATURE NAMES
    # =========================================================================
    edge_feature_names = [
        "dist",
        "align_i",
        "align_j",
        "lin_diff",
        "scat_avg",
        "z_diff",
        "z_min",
        "directional_agreement",
        "vertical_alignment",
        "thickness_diff",
        "verticality_diff"
    ]
    edge_feature_names = edge_feature_names[:n_edge_features]
    while len(edge_feature_names) < n_edge_features:
        edge_feature_names.append(f"edge_feature_{len(edge_feature_names)}")

    node_feature_names = [
        "thickness",
        "verticality",
        "linearity",
        "planarity",
        "scattering",
        "height",
        "eigenvalue_ratio",
        "omnivariance",
        "height_variation"
    ]
    node_feature_names = node_feature_names[:n_node_features]
    while len(node_feature_names) < n_node_features:
        node_feature_names.append(f"node_feature_{len(node_feature_names)}")

    # =========================================================================
    # STREAMING STATISTICS — EDGE FEATURES
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING EDGE FEATURE STATISTICS")
        print(f"{'='*80}")

    edge_stats = OnlineStats()
    for edge_features, _, _ in load_graph_data_generator(pt_files, batch_size=10000, verbose=verbose):
        edge_stats.update(edge_features)

    # =========================================================================
    # STREAMING STATISTICS — NODE FEATURES  (THE MISSING PIECE)
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("COMPUTING NODE FEATURE STATISTICS")
        print(f"{'='*80}")

    node_stats = OnlineStats()
    for pt_file in tqdm(pt_files, desc="Computing node stats", disable=not verbose):
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            node_features = graph_data['x'].numpy()
            if len(node_features) > 0:
                node_stats.update(node_features)
        except Exception as e:
            if verbose:
                print(f"\nError loading {pt_file}: {e}")
            continue

    # =========================================================================
    # QUANTILES — BOTH EDGE AND NODE
    # =========================================================================
    if verbose:
        print("\nComputing quantiles via reservoir sampling...")

    edge_quantiles, node_quantiles = compute_quantiles_streaming(
        pt_files,
        percentiles=[25, 50, 75],
        n_edge_features=n_edge_features,
        n_node_features=n_node_features,
        sample_size=min(100000, total_edges),
        verbose=verbose
    )

    # =========================================================================
    # BUILD FEATURE STATS DICTS
    # =========================================================================
    edge_feature_stats = {}
    for i, name in enumerate(edge_feature_names):
        edge_feature_stats[name] = {
            'min':    float(edge_stats.min_val[i]),
            'max':    float(edge_stats.max_val[i]),
            'mean':   float(edge_stats.mean[i]),
            'std':    float(edge_stats.std[i]),
            'median': float(edge_quantiles[50][i]),
            'q25':    float(edge_quantiles[25][i]),
            'q75':    float(edge_quantiles[75][i]),
            'range':  float(edge_stats.max_val[i] - edge_stats.min_val[i])
        }

    node_feature_stats = {}
    for i, name in enumerate(node_feature_names):
        node_feature_stats[name] = {
            'min':    float(node_stats.min_val[i]),
            'max':    float(node_stats.max_val[i]),
            'mean':   float(node_stats.mean[i]),
            'std':    float(node_stats.std[i]),
            'median': float(node_quantiles[50][i]),
            'q25':    float(node_quantiles[25][i]),
            'q75':    float(node_quantiles[75][i]),
            'range':  float(node_stats.max_val[i] - node_stats.min_val[i])
        }

    if verbose:
        print(f"\n{'='*80}")
        print("EDGE FEATURE SCALING STATISTICS")
        print(f"{'='*80}")
        print(f"{'Feature':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Median':>10}")
        print("-" * 80)
        for name in edge_feature_names:
            s = edge_feature_stats[name]
            print(f"{name:<25} {s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} "
                  f"{s['std']:>10.4f} {s['median']:>10.4f}")

        print(f"\n{'='*80}")
        print("NODE FEATURE SCALING STATISTICS")
        print(f"{'='*80}")
        print(f"{'Feature':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Median':>10}")
        print("-" * 80)
        for name in node_feature_names:
            s = node_feature_stats[name]
            print(f"{name:<25} {s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} "
                  f"{s['std']:>10.4f} {s['median']:>10.4f}")

    # =========================================================================
    # SAVE TO JSON  — now with separate node_* and edge_* scaling keys
    # =========================================================================
    scaling_params = None
    if save_stats:
        scaling_params = {
            # Metadata
            'edge_feature_names': edge_feature_names,
            'node_feature_names': node_feature_names,
            'n_edge_features': n_edge_features,
            'n_node_features': n_node_features,
            'n_edges': total_edges,
            'n_nodes': total_nodes,
            'n_graphs': len(pt_files),
            'graph_stats': {
                'edges_per_graph': {
                    'mean':   float(np.mean(edges_per_graph)),
                    'median': float(np.median(edges_per_graph)),
                    'min':    int(np.min(edges_per_graph)),
                    'max':    int(np.max(edges_per_graph))
                },
                'nodes_per_graph': {
                    'mean':   float(np.mean(nodes_per_graph)),
                    'median': float(np.median(nodes_per_graph)),
                    'min':    int(np.min(nodes_per_graph)),
                    'max':    int(np.max(nodes_per_graph))
                },
                'single_class_graphs': single_class_graphs,
                'mixed_class_graphs':  mixed_class_graphs,
                'class_0_only_graphs': class_0_only,
                'class_1_only_graphs': class_1_only
            },

            # ---- EDGE scaling (12 values each) ----
            'edge_standard_scaling': {
                'means': [edge_feature_stats[n]['mean'] for n in edge_feature_names],
                'stds':  [edge_feature_stats[n]['std']  for n in edge_feature_names]
            },
            'edge_minmax_scaling': {
                'mins': [edge_feature_stats[n]['min'] for n in edge_feature_names],
                'maxs': [edge_feature_stats[n]['max'] for n in edge_feature_names]
            },
            'edge_robust_scaling': {
                'medians': [edge_feature_stats[n]['median'] for n in edge_feature_names],
                'q25s':    [edge_feature_stats[n]['q25']    for n in edge_feature_names],
                'q75s':    [edge_feature_stats[n]['q75']    for n in edge_feature_names]
            },

            # ---- NODE scaling (10 values each) ----
            'node_standard_scaling': {
                'means': [node_feature_stats[n]['mean'] for n in node_feature_names],
                'stds':  [node_feature_stats[n]['std']  for n in node_feature_names]
            },
            'node_minmax_scaling': {
                'mins': [node_feature_stats[n]['min'] for n in node_feature_names],
                'maxs': [node_feature_stats[n]['max'] for n in node_feature_names]
            },
            'node_robust_scaling': {
                'medians': [node_feature_stats[n]['median'] for n in node_feature_names],
                'q25s':    [node_feature_stats[n]['q25']    for n in node_feature_names],
                'q75s':    [node_feature_stats[n]['q75']    for n in node_feature_names]
            },

            # Detailed per-feature stats
            'edge_feature_statistics': edge_feature_stats,
            'node_feature_statistics': node_feature_stats,
        }
        
        output_file = edges_dir / f'scaling_params_{split}.json'
        with open(output_file, 'w') as f:
            json.dump(scaling_params, f, indent=2)
        
        if verbose:
            print(f"\nScaling parameters saved to: {output_file}")
            print(f"  Edge scaling keys: edge_standard_scaling, edge_minmax_scaling, edge_robust_scaling")
            print(f"  Node scaling keys: node_standard_scaling, node_minmax_scaling, node_robust_scaling")

    # =========================================================================
    # CLASS BALANCE
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("CLASS BALANCE ANALYSIS")
        print(f"{'='*80}")

    class_0_count = 0
    class_1_count = 0
    for pt_file in tqdm(pt_files, desc="Counting classes", disable=not verbose):
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            labels = graph_data['y'].numpy()
            class_0_count += int(np.sum(labels == 0))
            class_1_count += int(np.sum(labels == 1))
        except Exception as e:
            continue

    total_samples = class_0_count + class_1_count
    if verbose and total_samples > 0:
        print(f"Class 0 (different trees): {class_0_count:,} ({class_0_count/total_samples*100:.2f}%)")
        print(f"Class 1 (same tree):       {class_1_count:,} ({class_1_count/total_samples*100:.2f}%)")
        if min(class_0_count, class_1_count) > 0:
            print(f"Imbalance ratio: {max(class_0_count, class_1_count)/min(class_0_count, class_1_count):.2f}:1")

    # =========================================================================
    # FEATURE IMPORTANCE (RF on edge features)
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS (edge features)")
        print(f"{'='*80}")

    sample_features = []
    sample_labels = []
    samples_collected = 0

    for pt_file in tqdm(pt_files, desc="Collecting RF sample", disable=not verbose):
        if samples_collected >= max_samples_rf:
            break
        try:
            graph_data = torch.load(pt_file, map_location='cpu', weights_only=True)
            edge_features = graph_data['edge_attr'].numpy()
            edge_labels = graph_data['y'].numpy()
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
            continue

    if sample_features:
        X_sample = np.vstack(sample_features)
        y_sample = np.concatenate(sample_labels)
        if verbose:
            print(f"Training Random Forest on {len(X_sample):,} samples...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_sample, y_sample)
        cv_scores = cross_val_score(rf, X_sample, y_sample, cv=3, scoring='f1')
        if verbose:
            print(f"Cross-val F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print("\nFeature importances:")
            for name, imp in sorted(zip(edge_feature_names, rf.feature_importances_),
                                    key=lambda x: x[1], reverse=True):
                print(f"  {name:<30}: {imp:.4f}")

    # =========================================================================
    # CLASS SEPARABILITY (edge features)
    # =========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("FEATURE SEPARABILITY ANALYSIS (edge features)")
        print(f"{'='*80}")

    class_0_stats, class_1_stats, c0_count, c1_count = compute_class_statistics_streaming(
        pt_files, n_edge_features, verbose
    )

    if verbose:
        print(f"\n{'Feature':<30} {'Class 0':>12} {'Class 1':>12} {'Diff':>10} {'Cohen d':>10}")
        print("-" * 78)
        for i, name in enumerate(edge_feature_names):
            m0 = class_0_stats.mean[i]
            m1 = class_1_stats.mean[i]
            diff = abs(m0 - m1)
            pooled_std = np.sqrt((class_0_stats.variance[i] + class_1_stats.variance[i]) / 2)
            d = diff / (pooled_std + 1e-10)
            print(f"  {name:<30} {m0:>12.4f} {m1:>12.4f} {diff:>10.4f} {d:>10.4f}")

    return edge_feature_stats, node_feature_stats, scaling_params


def main():
    edges_dir = Path("data/edges")
    
    for split in ['train', 'val', 'test']:
        if (edges_dir / split).exists():
            print(f"\n{'='*80}")
            print(f"Processing {split} split...")
            print(f"{'='*80}")
            
            analyze_graph_data(
                edges_dir,
                split,
                save_stats=True,
                verbose=True,
                max_samples_rf=50000
            )


if __name__ == "__main__":
    main()