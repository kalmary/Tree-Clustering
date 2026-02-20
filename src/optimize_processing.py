import numpy as np
import torch
import pathlib as pth
import optuna
import logging
from pprint import pprint
from typing import Union

from array_processing import TreeSegmGNN
from utils.instance_segmentation_evaluation import evaluate_segmentation


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_thresholds(
    model_name: str,
    data_files: list[pth.Path],
    output_probs: bool = False,
    edge_threshold: float = False,
    high_threshold: float = False,
    device: torch.device = torch.device('cuda'),
    radius: float = 1.5,
    voxel_factor: float = 0.78,
    max_nodes: int = 600,
    verbose: bool = False,
) -> dict:
    """
    Run segmentation on all .npy files and return aggregated metrics.
    Each file is expected to have XYZ in [:, :3] and ground-truth labels in [:, -1].
    """
    segmenter = TreeSegmGNN(
        model_name=model_name,
        device=device,
        use_mp=True,
        radius=radius,
        voxel_factor=voxel_factor,
        max_nodes=max_nodes,
        output_probs=output_probs,
        edge_threshold=edge_threshold,
        high_threshold=high_threshold,
        verbose=verbose,
    )

    all_metrics = []

    for file_path in data_files:
        cloud = np.load(file_path)
        xyz = cloud[:, :3]
        gt_labels = cloud[:, -1].astype(np.int32)

        pred_labels = segmenter.segment(xyz)
        metrics = evaluate_segmentation(pred_labels, gt_labels)
        all_metrics.append(metrics)

        logger.debug(f"{file_path.name}: PQ={metrics['PQ']:.4f}")

    del segmenter

    # Aggregate metrics across all files (mean)
    aggregated = {}
    for key in all_metrics[0].keys():
        aggregated[key] = float(np.mean([m[key] for m in all_metrics]))

    return aggregated


def objective(
    trial: optuna.Trial,
    model_name: str,
    data_files: list[pth.Path],
    device: torch.device,
    radius: float,
    voxel_factor: float,
    max_nodes: int,
) -> float:
    edge_threshold = trial.suggest_float("edge_threshold", 0.2, 0.7, step=0.05)
    high_threshold = trial.suggest_float("high_threshold", 0.4, 0.9, step=0.05)
    output_probs = trial.suggest_categorical("output_probs", [False, True])

    # high_threshold must be >= edge_threshold — prune invalid combos
    if high_threshold <= edge_threshold:
        raise optuna.exceptions.TrialPruned()

    logger.info(f"Trial {trial.number}: edge_threshold={edge_threshold:.2f}, high_threshold={high_threshold:.2f}") 

    metrics = evaluate_thresholds(
        model_name=model_name,
        data_files=data_files,
        output_probs=output_probs,
        edge_threshold=edge_threshold,
        high_threshold=high_threshold,
        device=device,
        radius=radius,
        voxel_factor=voxel_factor,
        max_nodes=max_nodes,
    )

    pq = metrics["PQ"]
    logger.info(f"Trial {trial.number} → PQ={pq:.4f} | metrics: {metrics}")

    return pq


def optimize_thresholds(
    model_name: str,
    data_dir: Union[str, pth.Path],
    n_trials: int = 50,
    device_name: str = "cpu",
    radius: float = 1.5,
    voxel_factor: float = 0.78,
    max_nodes: int = 600,
    study_name: str = "threshold_optimization",
    storage: str = "sqlite:///threshold_study.db",
) -> dict:
    """
    Main entry point. Runs Optuna study to find best edge_threshold and high_threshold.

    Args:
        model_name:   Model name (without .pt), passed to TreeSegmGNN.
        data_dir:     Directory containing .npy files (each with XYZ + GT label column).
        n_trials:     Number of Optuna trials.
        device_name:  'cpu', 'cuda', or 'gpu'.
        radius:       Radius for edge building (fixed during optimization).
        voxel_factor: Voxel factor (fixed during optimization).
        max_nodes:    Max nodes per subgraph (fixed during optimization).
        study_name:   Optuna study name.
        storage:      SQLite storage path for resumable studies.

    Returns:
        dict with best_edge_threshold, best_high_threshold, best_PQ, all best_params.
    """
    device = (
        torch.device("cuda")
        if ("cuda" in device_name.lower() or "gpu" in device_name.lower())
        and torch.cuda.is_available()
        else torch.device("cpu")
    )
    logger.info(f"Using device: {device}")

    data_dir = pth.Path(data_dir)
    data_files = sorted(data_dir.glob("*.npy"))
    assert len(data_files) > 0, f"No .npy files found in {data_dir}"
    logger.info(f"Found {len(data_files)} .npy files in {data_dir}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,         # allows resuming a previous run
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_name=model_name,
            data_files=data_files,
            device=device,
            radius=radius,
            voxel_factor=voxel_factor,
            max_nodes=max_nodes,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    result = {
        "best_edge_threshold": best.params["edge_threshold"],
        "best_high_threshold": best.params["high_threshold"],
        "best_PQ": best.value,
        "best_params": best.params,
    }

    print("\n" + "=" * 50)
    print("Optimization complete!")
    print(f"Best PQ:              {result['best_PQ']:.4f}")
    print(f"Best edge_threshold:  {result['best_edge_threshold']:.2f}")
    print(f"Best high_threshold:  {result['best_high_threshold']:.2f}")
    print("=" * 50)
    pprint(result)

    return result


if __name__ == "__main__":
    result = optimize_thresholds(
        model_name="EdgeGNN_2",
        data_dir="data/split/train",
        n_trials=50,
        device_name="cuda",
    )