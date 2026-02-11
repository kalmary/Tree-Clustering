import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field


@dataclass
class BalancingStats:
    """Accumulated statistics for the balancing operation."""
    files_processed: int = 0
    files_modified: int = 0
    files_removed: int = 0
    edges_removed: int = 0  # Changed from samples_removed
    single_class_files_removed: int = 0
    single_class_files_merged: int = 0
    highly_imbalanced_count: int = 0
    original_counts: dict = field(default_factory=lambda: {0: 0, 1: 0})
    final_counts: dict = field(default_factory=lambda: {0: 0, 1: 0})
    removed_per_file: list = field(default_factory=list)


def get_unique_edges_mask(edge_index: torch.Tensor) -> torch.Tensor:
    """
    For an undirected graph, get mask for unique edges (where source <= target).
    This avoids counting each edge twice.
    """
    return edge_index[0] <= edge_index[1]


def find_edge_pairs(edge_index: torch.Tensor, unique_indices: torch.Tensor) -> torch.Tensor:
    """
    For each unique edge index, find its corresponding reverse edge index.
    Returns tensor of shape (len(unique_indices), 2) where [:, 0] is forward, [:, 1] is reverse.
    """
    pairs = []
    
    for idx in unique_indices:
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        
        # Find reverse edge (v, u)
        if u == v:
            # Self-loop, no reverse
            pairs.append([idx, idx])
        else:
            reverse_mask = (edge_index[0] == v) & (edge_index[1] == u)
            reverse_idx = torch.where(reverse_mask)[0]
            
            if len(reverse_idx) > 0:
                pairs.append([idx, reverse_idx[0].item()])
            else:
                # No reverse found (shouldn't happen for undirected graphs)
                pairs.append([idx, idx])
    
    return torch.tensor(pairs, dtype=torch.long)


def balance_graph_data(
    graph_data: dict,
    target_ratio: float,
    majority_class: int,
    minority_class: int,
    min_edges_per_file: int,
    remove_isolated_nodes: bool = True
) -> Tuple[Optional[dict], int, int, int]:
    """
    Balance a single graph by removing majority class edges.
    
    Returns:
        (balanced_graph_data or None, edges_removed, new_class_0_count, new_class_1_count)
    """
    edge_index = graph_data['edge_index']
    edge_attr = graph_data['edge_attr']
    edge_labels = graph_data['y']
    node_features = graph_data['x']
    pos = graph_data['pos']
    
    # Get unique edges (to avoid double-counting in undirected graph)
    unique_mask = get_unique_edges_mask(edge_index)
    unique_indices = torch.where(unique_mask)[0]
    unique_labels = edge_labels[unique_indices]
    
    # Count classes on unique edges
    majority_count = (unique_labels == majority_class).sum().item()
    minority_count = (unique_labels == minority_class).sum().item()
    
    if majority_count == 0:
        return None, 0, 0, 0
    
    # Calculate target
    target_majority = int(minority_count * target_ratio)
    target_majority = max(target_majority, 1) if minority_count > 0 else 0
    
    if target_majority >= majority_count:
        # Already balanced
        return graph_data, 0, minority_count if minority_class == 0 else majority_count, \
               majority_count if minority_class == 1 else minority_count
    
    edges_to_remove = majority_count - target_majority
    
    # Separate unique edges by class
    majority_unique_mask = unique_labels == majority_class
    minority_unique_mask = unique_labels == minority_class
    
    majority_unique_indices = unique_indices[majority_unique_mask]
    minority_unique_indices = unique_indices[minority_unique_mask]
    
    # Randomly select majority edges to keep
    torch.manual_seed(42)
    perm = torch.randperm(len(majority_unique_indices))
    keep_majority_unique_indices = majority_unique_indices[perm[:target_majority]]
    
    # Combine with all minority edges
    keep_unique_indices = torch.cat([minority_unique_indices, keep_majority_unique_indices])
    
    # Find edge pairs (forward and reverse) for kept edges
    edge_pairs = find_edge_pairs(edge_index, keep_unique_indices)
    
    # Flatten to get all indices to keep (both directions)
    keep_all_indices = edge_pairs.flatten().unique()
    
    # Check if resulting graph would be too small
    if len(keep_all_indices) < min_edges_per_file:
        return None, majority_count, 0, 0
    
    # Slice edges
    new_edge_index = edge_index[:, keep_all_indices]
    new_edge_attr = edge_attr[keep_all_indices]
    new_edge_labels = edge_labels[keep_all_indices]
    
    # Optionally remove isolated nodes and reindex
    if remove_isolated_nodes:
        # Find unique nodes in the new edge index
        unique_nodes = new_edge_index.unique()
        num_new_nodes = len(unique_nodes)
        
        # Create mapping from old to new indices
        node_mapping = torch.full((node_features.size(0),), -1, dtype=torch.long)
        node_mapping[unique_nodes] = torch.arange(num_new_nodes)
        
        # Reindex edges
        new_edge_index = node_mapping[new_edge_index]
        
        # Slice node features and positions
        new_node_features = node_features[unique_nodes]
        new_pos = pos[unique_nodes]
    else:
        new_node_features = node_features
        new_pos = pos
        num_new_nodes = node_features.size(0)
    
    # Create new graph
    new_graph = {
        'x': new_node_features,
        'edge_index': new_edge_index,
        'edge_attr': new_edge_attr,
        'y': new_edge_labels,
        'pos': new_pos,
        'num_nodes': num_new_nodes
    }
    
    # Calculate final counts (on unique edges)
    final_unique_mask = get_unique_edges_mask(new_edge_index)
    final_unique_labels = new_edge_labels[final_unique_mask]
    final_class_0 = (final_unique_labels == 0).sum().item()
    final_class_1 = (final_unique_labels == 1).sum().item()
    
    return new_graph, edges_to_remove, final_class_0, final_class_1


def balance_graph_files(
    edges_dir: Union[str, Path],
    target_ratio: float = 1.0,
    split: str = 'train',
    min_edges_per_file: int = 100,
    dry_run: bool = False,
    backup: bool = True,
    verbose: bool = True,
    handle_single_class: str = 'remove',
    imbalance_threshold: float = 0.95,
    remove_isolated_nodes: bool = True
) -> dict:
    """
    Balance extremely imbalanced binary labeled graph data in .pt files.
    
    Args:
        edges_dir: Directory containing .pt files
        target_ratio: Target ratio of majority:minority (1.0 = balanced)
        split: Split identifier in directory structure
        min_edges_per_file: Minimum edges to keep a file
        dry_run: If True, only report without modifying
        backup: If True, create .bak copies
        verbose: If True, print progress
        handle_single_class: 'remove', 'merge', or 'keep'
        imbalance_threshold: Threshold for flagging imbalanced files
        remove_isolated_nodes: Remove nodes with no edges after balancing
        
    Returns:
        Dictionary of statistics
    """
    edges_dir = Path(edges_dir) / split
    
    if not edges_dir.exists():
        raise ValueError(f"Directory does not exist: {edges_dir}")
    
    pattern = "*.pt"
    pt_files = sorted(edges_dir.glob(pattern))
    
    if not pt_files:
        if verbose:
            print(f"WARNING: No .pt files found matching pattern '{pattern}' in {edges_dir}")
        return {}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"BALANCE GRAPH FILES - {split.upper()} SPLIT")
        print(f"{'='*80}\n")
        print(f"Found {len(pt_files)} files matching pattern '{pattern}'")
    
    stats = BalancingStats()
    
    # PHASE 0-1: Scan files and gather statistics
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 0-1: Scanning files and gathering statistics...")
        print(f"{'='*80}\n")
    
    file_info = []
    single_class_0_files = []
    single_class_1_files = []
    highly_imbalanced_files = []
    
    pbar = tqdm(pt_files, desc="Scanning files", disable=not verbose)
    
    for file_path in pbar:
        try:
            # Load graph (efficient loading with weights_only for security)
            graph_data = torch.load(file_path, map_location='cpu', weights_only=True)
            
            if 'edge_index' not in graph_data or 'y' not in graph_data:
                continue
            
            edge_labels = graph_data['y']
            edge_index = graph_data['edge_index']
            
            # Count unique edges (since graph is undirected)
            unique_mask = get_unique_edges_mask(edge_index)
            unique_labels = edge_labels[unique_mask]
            n_unique_edges = len(unique_labels)
            
            # Calculate counts
            class_0_count = (unique_labels == 0).sum().item()
            class_1_count = (unique_labels == 1).sum().item()
            
            class_0_pct = class_0_count / n_unique_edges if n_unique_edges > 0 else 0
            class_1_pct = class_1_count / n_unique_edges if n_unique_edges > 0 else 0
            
            only_class_0 = class_1_count == 0
            only_class_1 = class_0_count == 0
            highly_imbalanced = (class_0_pct > imbalance_threshold or
                               class_1_pct > imbalance_threshold)
            
            # Store minimal info
            info = {
                'path': file_path,
                'n_unique_edges': n_unique_edges,
                'class_0_count': class_0_count,
                'class_1_count': class_1_count,
                'class_0_pct': class_0_pct,
                'class_1_pct': class_1_pct,
                'only_class_0': only_class_0,
                'only_class_1': only_class_1,
                'highly_imbalanced': highly_imbalanced
            }
            
            # Categorize
            if only_class_0:
                single_class_0_files.append(info)
            elif only_class_1:
                single_class_1_files.append(info)
            elif highly_imbalanced:
                highly_imbalanced_files.append(info)
                file_info.append(info)
            else:
                file_info.append(info)
            
            # Accumulate counts
            stats.original_counts[0] += class_0_count
            stats.original_counts[1] += class_1_count
            stats.files_processed += 1
            
            # Clear memory
            del graph_data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            if verbose:
                print(f"WARNING: Error reading {file_path.name}: {e}")
            continue
    
    stats.highly_imbalanced_count = len(highly_imbalanced_files)
    
    # Report variance issues
    if verbose:
        total_single_class = len(single_class_0_files) + len(single_class_1_files)
        if total_single_class > 0 or highly_imbalanced_files:
            print(f"CLASS VARIANCE ISSUES DETECTED:")
            print(f"  Files with only class 0: {len(single_class_0_files)}")
            print(f"  Files with only class 1: {len(single_class_1_files)}")
            print(f"  Highly imbalanced files (>{imbalance_threshold*100:.0f}% one class): {len(highly_imbalanced_files)}")
            print(f"  Reasonably balanced files: {len(file_info) - len(highly_imbalanced_files)}")
            print(f"\nStrategy for single-class files: {handle_single_class.upper()}")
        else:
            print(f"No class variance issues detected")
            print(f"All {len(file_info)} files have reasonable class balance")
    
    # Handle single-class files
    single_class_files = single_class_0_files + single_class_1_files
    
    if handle_single_class == 'remove' and single_class_files:
        if verbose:
            print(f"\nRemoving {len(single_class_files)} single-class files...")
        
        pbar = tqdm(single_class_files, desc="Removing files", disable=not verbose)
        for info in pbar:
            if not dry_run:
                if backup:
                    backup_path = info['path'].with_suffix('.pt.bak')
                    info['path'].rename(backup_path)
                else:
                    info['path'].unlink()
            stats.single_class_files_removed += 1
    
    elif handle_single_class == 'merge' and single_class_files:
        if verbose:
            print(f"\nMerging single-class files...")
        
        # Merge class 0 graphs
        if single_class_0_files:
            merged_count = 0
            current_batch = []
            current_edge_count = 0
            max_edges = 5000
            
            pbar = tqdm(single_class_0_files, desc="Merging class 0", disable=not verbose)
            for info in pbar:
                graph_data = torch.load(info['path'], map_location='cpu', weights_only=True)
                n_edges = graph_data['edge_index'].size(1)
                
                if current_edge_count + n_edges <= max_edges:
                    current_batch.append(graph_data)
                    current_edge_count += n_edges
                else:
                    if current_batch and not dry_run:
                        merged_graph = merge_graphs(current_batch)
                        new_path = edges_dir / f"merged_class0_{merged_count:04d}.pt"
                        torch.save(merged_graph, new_path)
                        merged_count += 1
                    current_batch = [graph_data]
                    current_edge_count = n_edges
                
                del graph_data
            
            # Save remaining
            if current_batch and not dry_run:
                merged_graph = merge_graphs(current_batch)
                new_path = edges_dir / f"merged_class0_{merged_count:04d}.pt"
                torch.save(merged_graph, new_path)
                merged_count += 1
            
            if verbose:
                print(f"  Created {merged_count} merged class-0 files")
            
            # Remove originals
            if not dry_run:
                for info in single_class_0_files:
                    if backup:
                        info['path'].rename(info['path'].with_suffix('.pt.bak'))
                    else:
                        info['path'].unlink()
        
        # Merge class 1 graphs (similar logic)
        if single_class_1_files:
            merged_count = 0
            current_batch = []
            current_edge_count = 0
            max_edges = 5000
            
            pbar = tqdm(single_class_1_files, desc="Merging class 1", disable=not verbose)
            for info in pbar:
                graph_data = torch.load(info['path'], map_location='cpu', weights_only=True)
                n_edges = graph_data['edge_index'].size(1)
                
                if current_edge_count + n_edges <= max_edges:
                    current_batch.append(graph_data)
                    current_edge_count += n_edges
                else:
                    if current_batch and not dry_run:
                        merged_graph = merge_graphs(current_batch)
                        new_path = edges_dir / f"merged_class1_{merged_count:04d}.pt"
                        torch.save(merged_graph, new_path)
                        merged_count += 1
                    current_batch = [graph_data]
                    current_edge_count = n_edges
                
                del graph_data
            
            # Save remaining
            if current_batch and not dry_run:
                merged_graph = merge_graphs(current_batch)
                new_path = edges_dir / f"merged_class1_{merged_count:04d}.pt"
                torch.save(merged_graph, new_path)
                merged_count += 1
            
            if verbose:
                print(f"  Created {merged_count} merged class-1 files")
            
            # Remove originals
            if not dry_run:
                for info in single_class_1_files:
                    if backup:
                        info['path'].rename(info['path'].with_suffix('.pt.bak'))
                    else:
                        info['path'].unlink()
        
        stats.single_class_files_merged = len(single_class_files)
    
    if not file_info:
        if verbose:
            print("No valid files to process after variance check")
        return stats.__dict__
    
    # Recalculate counts from remaining files
    remaining_counts = {0: 0, 1: 0}
    for info in file_info:
        remaining_counts[0] += info['class_0_count']
        remaining_counts[1] += info['class_1_count']
    
    # Determine minority/majority
    minority_class = 0 if remaining_counts[0] < remaining_counts[1] else 1
    majority_class = 1 - minority_class
    minority_count = remaining_counts[minority_class]
    majority_count = remaining_counts[majority_class]
    
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 2: Analyzing label distribution...")
        print(f"{'='*80}\n")
        print(f"Original distribution:")
        print(f"  Class {minority_class} (minority): {minority_count:,} edges")
        print(f"  Class {majority_class} (majority): {majority_count:,} edges")
        print(f"  Imbalance ratio: {majority_count/max(minority_count, 1):.2f}:1")
    
    # Calculate targets
    target_majority_count = int(minority_count * target_ratio)
    edges_to_remove = majority_count - target_majority_count
    
    if verbose:
        print(f"\nTarget distribution (ratio {target_ratio}:1):")
        print(f"  Class {minority_class}: {minority_count:,} edges (unchanged)")
        print(f"  Class {majority_class}: {target_majority_count:,} edges")
        print(f"  Edges to remove: {edges_to_remove:,}")
    
    if edges_to_remove <= 0:
        if verbose:
            print("\nData is already balanced or minority class is larger. No action needed.")
        stats.final_counts = stats.original_counts.copy()
        return stats.__dict__
    
    # PHASE 3: Balance files
    if verbose:
        print(f"\n{'='*80}")
        print(f"PHASE 3: {'Simulating' if dry_run else 'Balancing'} files...")
        print(f"{'='*80}\n")
    
    pbar = tqdm(file_info, desc="Balancing files", disable=not verbose)
    
    for info in pbar:
        majority_in_file = info['class_1_count'] if majority_class == 1 else info['class_0_count']
        minority_in_file = info['class_0_count'] if majority_class == 1 else info['class_1_count']
        
        if majority_in_file == 0:
            continue
        
        # Calculate proportional target
        proportion = majority_in_file / majority_count
        target_majority_in_file = int(target_majority_count * proportion)
        
        if minority_in_file > 0:
            target_majority_in_file = max(target_majority_in_file, 1)
        
        edges_to_remove_from_file = majority_in_file - target_majority_in_file
        
        if edges_to_remove_from_file <= 0:
            continue
        
        # Load and balance
        graph_data = torch.load(info['path'], map_location='cpu', weights_only=True)
        
        balanced_graph, removed, new_class_0, new_class_1 = balance_graph_data(
            graph_data,
            target_ratio,
            majority_class,
            minority_class,
            min_edges_per_file,
            remove_isolated_nodes
        )
        
        del graph_data
        
        if balanced_graph is None:
            # File too small after balancing
            if not dry_run:
                if backup:
                    info['path'].rename(info['path'].with_suffix('.pt.bak'))
                else:
                    info['path'].unlink()
            stats.files_removed += 1
            stats.edges_removed += majority_in_file
        else:
            if not dry_run:
                if backup:
                    backup_path = info['path'].with_suffix('.pt.bak')
                    torch.save(torch.load(info['path'], map_location='cpu', weights_only=True), backup_path)
                torch.save(balanced_graph, info['path'])
            
            stats.files_modified += 1
            stats.edges_removed += removed
            stats.removed_per_file.append(removed)
        
        del balanced_graph
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate final counts
    final_majority = majority_count - stats.edges_removed
    stats.final_counts = {
        minority_class: minority_count,
        majority_class: final_majority
    }
    
    # Final summary
    if verbose:
        print(f"\n{'='*80}")
        print(f"{'[DRY RUN] ' if dry_run else ''}SUMMARY")
        print(f"{'='*80}\n")
        print(f"Single-class files removed: {stats.single_class_files_removed}")
        print(f"Single-class files merged: {stats.single_class_files_merged}")
        print(f"Files modified: {stats.files_modified}")
        print(f"Files removed (too small): {stats.files_removed}")
        print(f"Total edges removed: {stats.edges_removed:,}")
        print(f"\nFinal distribution:")
        print(f"  Class {minority_class}: {minority_count:,}")
        print(f"  Class {majority_class}: {final_majority:,}")
        print(f"  Final ratio: {final_majority/max(minority_count, 1):.2f}:1")
        
        if dry_run:
            print("\nThis was a dry run. No files were modified.")
    
    return stats.__dict__


def merge_graphs(graphs: list) -> dict:
    """
    Merge multiple graphs into a single graph.
    Node indices are renumbered to be sequential.
    """
    if not graphs:
        return None
    
    if len(graphs) == 1:
        return graphs[0]
    
    merged_x = []
    merged_edge_index = []
    merged_edge_attr = []
    merged_y = []
    merged_pos = []
    
    node_offset = 0
    
    for graph in graphs:
        merged_x.append(graph['x'])
        merged_pos.append(graph['pos'])
        merged_edge_attr.append(graph['edge_attr'])
        merged_y.append(graph['y'])
        
        # Offset edge indices
        offset_edge_index = graph['edge_index'] + node_offset
        merged_edge_index.append(offset_edge_index)
        
        node_offset += graph['num_nodes']
    
    return {
        'x': torch.cat(merged_x, dim=0),
        'edge_index': torch.cat(merged_edge_index, dim=1),
        'edge_attr': torch.cat(merged_edge_attr, dim=0),
        'y': torch.cat(merged_y, dim=0),
        'pos': torch.cat(merged_pos, dim=0),
        'num_nodes': node_offset
    }


if __name__ == "__main__":
    edges_dir = Path("data/edges")
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        stats = balance_graph_files(
            edges_dir,
            target_ratio=1.0,
            split=split,
            dry_run=False,
            backup=False,
            verbose=True,
            handle_single_class='remove',
            remove_isolated_nodes=True
        )
