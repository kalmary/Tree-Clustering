import matplotlib
matplotlib.use('Agg')


import pathlib as pth
import numpy as np
from pprint import pprint
from typing import Union, Sequence
import itertools
import sys
import argparse
import logging
import datetime
import multiprocessing


import torch
import torch.nn as nn
from torchinfo import summary
import optuna
import shutil

from tqdm import tqdm


src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))


from _train_single_case import train_model
from utils import load_json, save2json, save_model, convert_str_values
from data_processing.preprocess_edges import preprocess_dataset
from data_processing.balance_edges import balance_edge_files
from utils import Plotter

from AffinityMLP import AffinityMLP


def check_models(model_configs_paths: list[pth.Path],
                 max_input_size = (1, 8192, 4),
                 max_memory_GB = 20,
                 verbose: bool = False) -> tuple[list[dict], list[pth.Path]]:
    """
    Check if models defined in NeuralNet/Architectures/models_1D/model_configs compile.
    Print models not compiling.
    Return list of model configs that compile.
"""
    logger = logging.getLogger(__name__)
    logger.info(f'START: check_models.')

    # str paths to pth.Path if neccessary
    model_configs_paths = [pth.Path(config) for config in model_configs_paths]

    # check each model if it compiles and take not more than max memory
    model_configs = []
    for index, model_config_path in enumerate(model_configs_paths.copy()):
        model_config = load_json(model_config_path)
        model_config = convert_str_values(model_config)

        try:
            model = AffinityMLP(model_config)
            model.eval()
            model_summary = summary(model, input_size=max_input_size, verbose=0)
            estimated_memory_GB = (model_summary.total_param_bytes + model_summary.total_output_bytes) / (1024 ** 3 )

            if estimated_memory_GB > max_memory_GB:
                    raise MemoryError(f"Estimated memory {estimated_memory_GB:.2f} GB exceeds limit of {max_memory_GB:.2f} GB.")

            del model, model_summary

            if verbose:
                logger.info(f'Model {model_config_path.name} compiled successfully. Estimated memory: {estimated_memory_GB:.2f} GB.')
                print(f"Model {model_config_path.name} compiled successfully\nEstimated memory: {estimated_memory_GB:.2f} GB.\n")

        except Exception as e:
            logger.error(f'Error compiling model {model_config_path.name}: {e}') 
            if verbose:
                print(f"Error compiling model {model_config_path.name}:\n{e}")
            model_configs_paths.pop(index)
        else:
            model_configs.append(model_config)
    logger.info(f'STOP: check_models. {len(model_configs)} models compiled successfully.') 
    
    return model_configs, model_configs_paths

def get_step_list(param_value_list: list[Union[int, float]]) -> list[Union[int, float]]:
    """"Generate a list of values based on the given parameter value and type of list elements."""

    start, stop, step = param_value_list
    
    if all(isinstance(x, int) for x in [start, stop, step]):
        return list(range(int(start), int(stop + step), int(step)))
    elif all(isinstance(x, (int, float)) for x in [start, stop, step]):
        return [float(x) for x in np.arange(float(start), float(stop + step), float(step))]
    else:
        raise ValueError(f"Invalid parameter values: {param_value_list}. Must be all int or all float.")

def get_factor_list(param_value_list: list[Union[float]]) -> list[Union[float]]:
    """"Generate a list of values based on the given parameter value of list elements."""

    start, stop, factor = param_value_list
    factor_list = []
    
    num = start
    while num >= stop:

        factor_list.append(num)
        num *= factor

    factor_list.sort()
    return factor_list

    

def generate_experiment_configs(training_config: dict, 
                                model_configs_list: Sequence[dict],
                                device_name: str = 'cpu') -> list[dict]:
    logger = logging.getLogger(__name__)
    logger.info(f'START: generate_experiment_config.')
    
    device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')
    logger.info(f'Using device: {device}')

    dynamic_params = {}
    static_params = {}

    # Separate dynamic and model, base_path=base_path, existing_ok=Truestatic parameters

    for key, value in training_config.items():
        
        if "comment" in key.lower():
            continue
        
        elif isinstance(value, list) and len(value) > 1 and not 'samples_len' in key.lower():
            if "learning_rate" in key.lower() or "weight_decay" in key.lower():
                dynamic_params[key] = get_factor_list(training_config[key])
            else:
                dynamic_params[key] = get_step_list(value)

        elif 'samples_len' in key.lower():
            samples_len = get_step_list(value)
            static_params[key] = samples_len
        else:
            static_params[key] = value

    static_params['device'] = device
    
    logger.info(f'Generated {len(dynamic_params)} dynamic parameters.')
    logger.info(f'Generated {len(static_params)} static parameters.')

    # Generate all combinations of dynamic parameters
    keys = dynamic_params.keys()
    values_lists = dynamic_params.values()
    combinations = list(itertools.product(*values_lists))

    
    # Create experiment configurations
    exp_configs = []
    for combo in combinations:
        combo_dict = dict(zip(keys, combo))
        combo_dict.update(static_params)

        for model_config in model_configs_list:
            
            # get model config
            dynamic_config = dict(zip(keys, combo))
            
            exp_config = static_params.copy()
            exp_config.update(dynamic_config)
            exp_config['model_config'] = model_config
            
            exp_configs.append(exp_config)
        
    logger.info(f'Generated {len(exp_configs)} experiment configurations.')
    logger.info(f'STOP: generate_experiment_config')


    return exp_configs



def load_config(base_dir: Union[str, pth.Path], device_name: str, mode: int = 0) -> list[dict]:

    """
    Load configuration files and prepare experiment configurations for training.
    mode:
    0 - single_training
    1 - multiple trainings, grid_based
    2 - multiple trainings, with optuna
    
    """
    logger = logging.getLogger(__name__)
    logger.info(f'START: case_based_training.')

    if isinstance(mode, int):
        if mode not in [0, 1, 2]:
            raise ValueError(f"Invalid mode: {mode}. Must be:\n" \
                             "0 - test" \
                             "1 - single_training," \
                             "2 - multiple trainings, with optuna.")

    base_dir = pth.Path(base_dir)
    config_files_dir = base_dir.joinpath('training_configs')
    model_configs_dir = base_dir.joinpath('model_configs')

    model_configs_paths_list = list(model_configs_dir.rglob('*.json'))
    logger.info(f'Found {len(model_configs_paths_list)} model configs in dir {model_configs_dir}')


    if mode == 0 or mode == 1:
        training_config = load_json(config_files_dir.joinpath('config_train_single.json'))
    elif mode == 2:
        training_config = load_json(config_files_dir.joinpath('config_train.json'))
    
    logger.info(f'Loaded training config for mode: {mode}.')

    if mode == 0 or mode == 1:
        model_configs_paths_list = [p for p in model_configs_paths_list if "single" in p.stem]
    else:
        model_configs_paths_list = [p for p in model_configs_paths_list if "single" not in p.stem]
    
    training_config = convert_str_values(training_config)
    model_configs_list, _ = check_models(model_configs_paths_list, max_input_size=(8, 2*8192, 4), max_memory_GB=32)
    
    assert model_configs_list != 0, "No models compiled. Check model_configs - most likely too big models are defined"

    if mode == 2:
        device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')
        training_config['device'] = device
        
        logger.info(f'Loaded device: {device}')
        logger.info(f'STOP: load_config. All files loaded.')

        training_config['model'] = None

        return [training_config, model_configs_list]
    else:
        exp_configs = generate_experiment_configs(training_config, 
                                            model_configs_list, 
                                            device_name = device_name)
        
        logger.info(f'STOP: load_config. All files loaded.')

        return exp_configs

def test_case(exp_config: dict) -> None:
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: test_case.')
    
    """Test case for training model with reduced parameters for quick execution."""
    exp_config['train_repeat'] = 2
    exp_config['learning_rate'] = 0.01
    exp_config['epochs'] = 2
    
    try:
        for _, _ in train_model(config=exp_config):
            pass
    except Exception as e:
        logger.error(f'ERROR: test_case. Error message: {e}')
        print(f"Error training model (TESTING_MODE):\n{e}")
    logger.info('STOP: test_case passed.')

class Checkpoint: 
    def __init__(self, existing_ok: bool = False) -> None:
        self.existing_ok = existing_ok
        self.final_val_best = 0.
        self.save_new = False
    
    def check_checkpoint(self, 
                         model: nn.Module,
                         model_name: str,
                         final_val: float,
                         exp_config: dict,
                         result_hist: dict) -> tuple[nn.Module, dict, pth.Path]:
    
        

        logger = logging.getLogger(__name__)

        model_dir =  pth.Path(__file__).parent.joinpath('training_results').joinpath(f'{model_name.rsplit('_', 1)[0]}')
        model_dir.mkdir(exist_ok=True, parents=True)

        model_path = model_dir.joinpath(f'{model_name}.pt')

        plot_dir = model_dir.joinpath('plots')
        if not plot_dir.exists():
            logger.info(f'Plots directory created: {plot_dir}')

        dict_files_dir = model_dir.joinpath('dict_files')
        if not dict_files_dir.exists():
            logger.info(f'Dict files directory created: {dict_files_dir}')
        
        plot_dir.mkdir(exist_ok=True, parents=True)
        if not plot_dir.exists():
            logger.info(f'Plots directory created: {plot_dir}')

        dict_files_dir.mkdir(exist_ok=True, parents=True)

        config_path = dict_files_dir.joinpath(f'{model_path.stem}_config.json')


        if final_val > self.final_val_best:
            self.save_new = True
            self.final_val_best = final_val
        
        if not self.save_new:
            return [model, exp_config, config_path]

        if not self.existing_ok:
                for file_path in plot_dir.iterdir():
                    file_path.unlink()
                for file_path in dict_files_dir.iterdir():
                    file_path.unlink()

        model_path = save_model(model_path, model, 
                                        existing_ok=self.existing_ok)
        model_name = model_path.stem
        logger.info(f'New best model saved to: {model_path}. model_name: {model_name}')


        best_config = exp_config
        best_config['device'] = str(best_config['device'])

        config_path = dict_files_dir.joinpath(f'{model_path.stem}_config.json')
        save2json(best_config, config_path)
        logger.info(f'New config for model {model_name} saved to: {config_path}')



        if len(result_hist['f1_hist']) > 1:
            plotter_obj = Plotter(0, plots_dir = plot_dir)


            plotter_obj.plot_metric_hist(f'Loss_{model_name}.png',
                                         result_hist['loss_hist'],
                                         result_hist['loss_v_hist'])

            plotter_obj.plot_metric_hist(f'F1_SCORE{model_name}.png', 
                                    result_hist['f1_hist'],
                                    result_hist['f1_v_hist'])

            logger.info(f'Metrics history plots for {model_name} saved to: {plot_dir}')


        self.save_new = False
        return model, best_config, config_path

        




def case_based_training(exp_configs: list[dict],
                        model_name: str) -> None:
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: case_based_training.')

    pbar = tqdm(enumerate(exp_configs), total=len(exp_configs), desc="Training models", unit="model", position=0, leave=False)

    model_dir =  pth.Path(__file__).parent.joinpath('training_results').joinpath(f'{model_name.rsplit('_', 1)[0]}')
    model_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Model directory created: {model_dir}')

    model_path = model_dir.joinpath(f'{model_name}.pt')
    logger.info(f'Model path created: {model_path}')

    plot_dir = model_dir.joinpath('plots')
    dict_files_dir = model_dir.joinpath('dict_files')
    

    plot_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Plots directory created: {plot_dir}')
    dict_files_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Dict files directory created: {dict_files_dir}')
    

    logger.info('Case based training starting.')
    checkpoint = Checkpoint(existing_ok=False)


    for i, exp_config in pbar:
        logger.info(f'Case {i+1}/{len(exp_configs)}: {exp_config}')

        for model, result_hist in train_model(config=exp_config):
            logger.info(f'Single model was generated. val_acc: {result_hist["acc_v_hist"][-1]:.3f}  val_loss: {result_hist["loss_v_hist"][-1]:.3f}')

            final_val = result_hist['acc_v_hist'][-1]*0.6 + (1 / (1 + result_hist['loss_v_hist'][-1]))*0.4

            model, best_config, config_path, result_hist = checkpoint.check_checkpoint(model, model_name, final_val, exp_config, result_hist)
    
    logger.info(f'Best model saved to: {model_path}')
    logger.info(f'Best config saved to: {config_path}')
    logger.info('STOP: case_based_training')

    summary(model)

def preprocess_data(radius: Union[float, list[float]]):
    if isinstance(radius, float):
        radius = [radius]

    logger = logging.getLogger(__name__)

    project_dir = pth.Path(__file__).parent.parent.parent
    edges_dir = project_dir.joinpath('data/edges')

    new = []
    for file in edges_dir.glob('*.txt'):
        with open(edges_dir.joinpath(file, 'r+')) as f:
            current_radius = f.read()
            for rad in radius:
                if float(current_radius) != rad:
                    f.seek(0)
                    f.write(str(rad))
                    f.truncate()
                    new.append(True)
                else:
                    new.append(False)
    update_dataset = any(new)
    if not update_dataset:
        return
    else:
        logger.info(f'Updated radius in {edges_dir.joinpath("radius.txt")} to {radius}')
    
        # clear edges dir but leave radius.txt
        for file in edges_dir.iterdir():
            if file.is_dir() and file.suffix != '.txt':
                shutil.rmtree(file)
                logger.info(f'Deleted existing edge directory: {file}')
        split_dir = edges_dir.parent.joinpath('split')

        logger.info(f'Preprocessing dataset with new radius values: {radius}')
        preprocess_dataset(
                input_dir=split_dir,
                output_dir=edges_dir,
                radius=radius,
                use_mp=True,
                verbose=False,
                n_jobs=-1
            )
        
        balance_edge_files(edges_dir=edges_dir,
                        target_ratio=2.,
                        split='train')
        

            

           


def objective_function(trial: optuna.Trial,
                       exp_config: list[dict], # exp config, converted from str
                       model_name: str,
                       model_configs_list: list[dict],
                       checkpoint: object) -> float:
    
    """
    Objective function for Optuna
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: objective_function')

    existing_ok = False


    model_config_index = trial.suggest_categorical('model_config_index', range(len(model_configs_list)))
    model_config = model_configs_list[model_config_index]


    lr = trial.suggest_categorical('learning_rate', get_factor_list(exp_config['learning_rate']))
    weight_decay = trial.suggest_categorical('weight_decay', get_factor_list(exp_config['weight_decay']))

    batch_size = trial.suggest_categorical('batch_size', get_step_list(exp_config['batch_size']))
    epochs = trial.suggest_categorical('epochs', get_step_list(exp_config['epochs']))

    radius = trial.suggest_categorical('radius', get_step_list(exp_config['radius']))

    pct_start = trial.suggest_categorical('pct_start', get_step_list(exp_config['pct_start']))
    div_factor = trial.suggest_categorical('div_factor', get_factor_list(exp_config['div_factor']))
    fin_div_factor = trial.suggest_categorical('final_div_factor', get_factor_list(exp_config['final_div_factor']))


    exp_config = exp_config.copy()
    exp_config.update({
        'model_config': model_config,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'radius': radius,
        'pct_start': pct_start,
        'div_factor': div_factor,
        'final_div_factor': fin_div_factor,
        'train_repeat':1
    })

    preprocess_data(radius=radius)


    logger.info(f'Generated exp_config for trial: {trial.number}')
    for key, value in exp_config.items():
        if key != 'model_config':
            logger.info(f'parameters: {key}: {value}')
    for key, value in model_config.items():
        logger.info(f'model_config: {key}: {value}')

    
    # start training
    best_val_f1 = 0.0
    best_val_loss = float('inf')

    for epoch_idx, (model, result_hist) in enumerate(train_model(config=exp_config)):
        
        if model is None or result_hist == {}:
            logger.error(f'ERROR: objective_function. Empty result_hist. Trial pruned.')
            trial.report(0.0, step=epoch_idx)
            raise optuna.exceptions.TrialPruned()

        # goal is to maximize val_acc + 1/val_loss
        final_val_f1 = result_hist['f1_v_hist'][-1]
        final_val_loss = result_hist['loss_v_hist'][-1]

        best_val_f1 = max(best_val_f1, final_val_f1)
        best_val_loss = min(best_val_loss, final_val_loss)


        normalized_loss = 1 / (1 + best_val_loss)
        final_val = 0.4 * best_val_f1 + 0.6 * normalized_loss

        checkpoint.check_checkpoint(model,
                                    model_name,
                                    final_val,
                                    exp_config,
                                    result_hist)

        logger.info(f'Epoch {epoch_idx+1}/{exp_config["epochs"]}: best_val_f1: {best_val_f1:.3f}, best_val_loss: {best_val_loss:.3f}, final_val: {final_val:.3f}')

        # report acutal results for prunning
        trial.report(final_val, step=epoch_idx)

        if trial.should_prune():
            logger.info(f'Pruning trial: {trial.number}')
            raise optuna.exceptions.TrialPruned()
        
    logger.info(f'STOP: objective_function')

    return final_val

def optuna_based_training(exp_config: list[dict], # only one, non converted conf given in list
                          model_name: str,
                          n_trials: int = 100) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    logger = logging.getLogger(__name__)
    logger.info(f'START: optuna_based_training.')

    # able to automatically stop poor working exps
    n_startup = 3
    n_warmup_steps = 15
    interval_steps = 5

    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup, n_warmup_steps=n_warmup_steps, interval_steps=interval_steps)
    logger.info(f'Pruner created: parameters: n_startup_trials: {n_startup}, n_warmup_step: {n_warmup_steps}, interval_steps: {interval_steps}')

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(),
        study_name = f'{model_name}_{timestamp}',
        storage=f'sqlite:///db.sqlite3',
        directions=['maximize'],
        pruner=pruner)
    logger.info(f'Study created. Check ')


    model_configs = exp_config[1]
    exp_config = exp_config[0]

    train_repeat_old = exp_config['train_repeat']
    
    # Create progress bar
    pbar = tqdm(total=n_trials, desc="Optuna Optimization", unit="trial")
    
    def callback(study, trial):
        pbar.update(1)
        try:
            pbar.set_postfix({
                "Trial": trial.number,
                "Best Value": f"{study.best_value:.4f}",
                "Current": f"{trial.value:.4f}" if trial.value else "Pruned"
            })
        except ValueError:
            pbar.set_postfix({"Status": "Pruned"})

    checkpoint = Checkpoint(existing_ok=False)                          

    # Single optimize call with callback
    study.optimize(lambda trial: objective_function(trial,
                                                    exp_config=exp_config,
                                                    model_name = model_name,
                                                    model_configs_list=model_configs,
                                                    checkpoint=checkpoint),
                   n_trials=n_trials,
                   callbacks=[callback])
    
    pbar.close()
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    logger.info(f'Optimization finished. Best value of formula: 0.6 * val_acc + 0.4 * norm_val_loss: {best_value:.4f}')

    print(20*'=')
    print(f'Optuna optimalization finished')
    print(f'Best value of 0.6 * val_acc + 0.4 * norm_val_loss: {best_value:.4f}')
    pprint(f'Best trial params:\n{best_params}')
    print(20*'=')


    best_model_config = model_configs[best_params.pop('model_config_index')] 

    final_exp_config = exp_config.copy()
    final_exp_config.update(best_params)
    final_exp_config.update({'train_repeat': train_repeat_old})


    final_exp_config['model_config'] = best_model_config

    print('Training the best model last time: ')

    case_based_training([final_exp_config],
                        model_name=model_name) # FIXME inproper dict creation
    
    logger.info(f'STOP: optuna_based_training')
    
def argparser():
        
    """
    Parse command-line arguments for automated CNN training pipeline configuration.
    Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """

    parser = argparse.ArgumentParser(
        description="Script for training the model based on predefined range of scenarios",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID"
        )
    )

    
    # Flag definition
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'gpu'], # choice limit
        help=(
            "Device for tensor based computation.\n"
            "Pick 'cpu' or 'cuda'/ 'gpu'.\n"
        )
    )

    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        choices=[0, 1, 2, 3], # choice limit
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: test\n'
            '1: single training\n'
            '2: multiple trainings, with optuna\n'
            '3: only check models'
        )
    )

    return parser.parse_args()

def update_paths(exp_configs: list) -> list:
    base_path = pth.Path(__file__).parent.parent.parent
    data_path = base_path.joinpath('data/edges')
    data_path.mkdir(exist_ok=True, parents=True)

    # Handle mode 2 case: [exp_config_dict, model_configs_list]
    if len(exp_configs) == 2 and isinstance(exp_configs[1], list):
        exp_config = exp_configs[0]
        exp_config["data_path_train"] = str(data_path.joinpath('train'))
        exp_config["data_path_val"] = str(data_path.joinpath('val'))
        exp_config["data_path_test"] = str(data_path.joinpath('test'))
        return exp_configs
    
    # Handle mode 1 case: list of exp_config dicts
    for exp_config in exp_configs:
        exp_config["data_path_train"] = str(data_path.joinpath('train'))
        exp_config["data_path_val"] = str(data_path.joinpath('val'))
        exp_config["data_path_test"] = str(data_path.joinpath('test'))

    return exp_configs

def main():
    multiprocessing.set_start_method('spawn', force=True)
    args = argparser()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = pth.Path(__file__).parent.joinpath('logs')
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file_name = f'{args.model_name}_{timestamp}_training.log'
    log_file_path = log_dir.joinpath(log_file_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"PROGRAM START: {args.__dict__}")


    device = args.device.lower()
    model_name = args.model_name

    base_path = pth.Path(__file__).parent
    if args.mode != 4:
        exp_configs  = load_config(base_path, device, mode = args.mode)
        exp_configs = update_paths(exp_configs)

    if args.mode == 0:
        test_case(exp_config=exp_configs[0])
    elif args.mode == 1 or args.mode==2:
        case_based_training(exp_configs=exp_configs,
                            model_name=model_name)
    elif args.mode == 3:
        optuna_based_training(exp_config=exp_configs,
                              model_name=model_name,
                              n_trials=80)
    elif args.mode == 4:
        model_configs_dir = base_path.joinpath('model_configs')
        model_configs_paths_list = list(model_configs_dir.rglob('*.json'))

        check_models(model_configs_paths=model_configs_paths_list, 
                     max_input_size=(1, 8192, 4), 
                     max_memory_GB=20,
                     verbose=True)

    logger.info("PROGRAM END")
        

if __name__ == '__main__':
    
    main()  