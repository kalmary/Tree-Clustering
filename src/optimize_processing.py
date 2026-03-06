import numpy as np
import torch
import pathlib as pth
import optuna
import logging
from pprint import pprint
from typing import Union
import tqdm
import random

from array_processing_RE import TreeSegmRay
from utils.instance_segmentation_evaluation import evaluate_segmentation
import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_thresholds(
    data_files: list[pth.Path],
    height_min: float,
    max_diameter: float,
    distance_limit: float,
    gravity_factor: float,
    use_rays: bool,
    verbose: bool = True,
) -> dict:
    """
    Run segmentation on all .npy files and return aggregated metrics.
    Each file is expected to have XYZ in [:, :3] and ground-truth labels in [:, -1].
    """
    segmenter = TreeSegmRay(height_min=height_min,
                            max_diameter=max_diameter,
                            distance_limit=distance_limit,
                            gravity_factor=gravity_factor,
                            use_rays=use_rays,
                            verbose = False)

    all_metrics = []
    
    pbar = data_files
    if verbose:
        pbar = tqdm.tqdm(data_files, desc="Processing files", position=0, leave=False)

    segmenter.start_container()
    for file_path in pbar:
        cloud = np.load(file_path)
        xyz = cloud[:, :3]
        gt_labels = cloud[:, -1].astype(np.int32)

        try:
            pred_labels = segmenter.segment(xyz)
        except Exception as e:
            logger.warning(f"Failed to segment {file_path.name}: {e}")
            return {}
        
        metrics = evaluate_segmentation(pred_labels, gt_labels)
        all_metrics.append(metrics)

        logger.debug(f"{file_path.name}: seg_quality={metrics['seg_quality']:.4f}")

    segmenter.rm_container()
    del segmenter

    # Aggregate metrics across all files (mean)
    aggregated = {}
    for key in all_metrics[0].keys():
        aggregated[key] = float(np.mean([m[key] for m in all_metrics]))

    return aggregated


def objective(
    trial: optuna.Trial,
    data_files: list[pth.Path],

) -> float:

    height_min = trial.suggest_float("height_min", 0.5, 2.0, step=0.25)
    max_diameter = trial.suggest_float("max_diameter", 0.5, 1.5, step=0.1)
    distance_limit = trial.suggest_float("distance_limit", 0.1, 2.0, step=0.05)
    gravity_factor = trial.suggest_float("gravity_factor", 0.1, 0.9, step=0.1)
    use_rays = trial.suggest_categorical("use_rays", [True, False])

    logger.info(f"Trial {trial.number}") 

    metrics = evaluate_thresholds(
        data_files=data_files,
        height_min=height_min,
        max_diameter=max_diameter,
        distance_limit=distance_limit,
        gravity_factor=gravity_factor,
        use_rays=use_rays,
        verbose=True
        
    )

    if len(metrics)==0:
        raise optuna.exceptions.TrialPruned()


    seq_q = metrics['seg_quality']
    logger.info(f"Trial {trial.number} → seg_quality={seq_q:.4f} | metrics: {metrics}")
    
    return seq_q


def optimize_thresholds(
    data_dir: Union[str, pth.Path],
    n_trials: int = 50,
    file_ratio: float = 0.4,
    study_name: str = "threshold_optimization_RE",
    storage: str = 'sqlite:///db.sqlite3',
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
        dict with best_edge_threshold, best_high_threshold, best_f1, all best_params.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    study_name = study_name + f"_{date}"

    data_dir = pth.Path(data_dir)
    data_files = sorted(data_dir.glob("*.npy"))
    random.shuffle(data_files)
    data_files = data_files[:int(file_ratio*len(data_files))]
    assert len(data_files) > 0, f"No .npy files found in {data_dir}"
    logger.info(f"Found {len(data_files)} .npy files in {data_dir}")

    n_startup = 5
    n_warmup_steps = 40
    interval_steps = 10

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,         # allows resuming a previous run
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials = n_startup,
                                           n_warmup_steps=n_warmup_steps,
                                           interval_steps=interval_steps),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            data_files=data_files,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    result = {
        "best_edge_threshold": best.params["edge_threshold"],
        "best_crown_threshold_reduction": best.params["crown_threshold_reduction"],
        "best_val": best.value,
        "best_params": best.params,
    }

    print("\n" + "=" * 50)
    print("Optimization complete!")
    print(f"Best value (seg_quality):              {result['best_val']:.4f}")
    print(f"Best edge_threshold:  {result['best_edge_threshold']:.2f}")
    print(f"Best crown_threshold_reduction:  {result['best_crown_threshold_reduction']:.2f}")
    print("=" * 50)
    pprint(result)

    return result


if __name__ == "__main__":
    result = optimize_thresholds(
        data_dir="data/split/test",
        n_trials=100
    )
