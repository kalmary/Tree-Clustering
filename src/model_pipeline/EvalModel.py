import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchinfo import summary

import argparse
from tqdm import tqdm
import pathlib as pth


from AffinityMLP import AffinityMLP
from _data_loader import EdgeDataset

import os
import sys

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils import load_json, load_model, convert_str_values
from utils import get_dataset_len
from utils.weights import calculate_binary_weights
from utils.metrics import binary_f1_score
from utils import get_intLabels
from utils import Plotter, ClassificationReport


def _eval_model(config_dict: dict,
                model: nn.Module) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    device_gpu = torch.device('cuda')
    device_cpu = torch.device('cpu')

    device_loader = device_cpu
    device_loss = device_cpu
    
    test_dataset = EdgeDataset(
        base_dir=config_dict['data_path_test'],
        batch_size=config_dict['batch_size'],
        shuffle=True,
        device=device_loader
    )
    
    testLoader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=10,
        pin_memory=False
    )

    model = AffinityMLP(config_dict['model_config'], scaling_config=config_dict['scaling_config'])

    total = get_dataset_len(testLoader, verbose=False)
    assert total > 0, "Testing dataset is empty."

    weights = calculate_binary_weights(testLoader, total=total, verbose=False)
    weights = weights.to(device_loss)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights).to(config_dict['device'])

    loss_per_epoch = 0.
    f1_per_epoch = 0.0
    epoch_samples = 0

    pbar = tqdm(testLoader, total=total, desc="Testing", unit="batch")
    with torch.no_grad():
        for batch_x, batch_y in pbar:

            model.eval()
            batch_x = batch_x.to(config_dict['device'])


            outputs = model(batch_x)
            outputs = outputs.to(device_loss)
            batch_y = batch_y.to(device_loss)

            loss = criterion(outputs, batch_y)

            probs = torch.sigmoid(outputs.cpu())
            labels = torch.argmax(outputs.cpu(), dim=1)

            f1 = binary_f1_score(probs, batch_y)
            
            loss_per_epoch += loss.item()*batch_y.size(0)
            f1_per_epoch += f1*batch_y.size(0)

            epoch_samples += batch_y.size(0)

            total_loss = loss_per_epoch / epoch_samples
            total_f1 = f1_per_epoch / epoch_samples

            pbar.set_postfix({
                'loss': f"{total_loss:.4f}",
                'f1': f"{total_f1:.4f}"
            })


    return total_loss, total_f1, all_probs, all_labels

def eval_model_front(config_dict: dict,
         model: nn.Module,
         paths: list[pth.Path]):

    model_path = paths[0]
    model_name = model_path.stem

    plot_dir = paths[1]

    total_loss, total_f1 = _eval_model(config_dict=config_dict, model=model)

    
    print('='*20)
    print('MODEL TESTED')
    print('Model path', model_path)
    print('Loss: ', total_loss)
    print('F1 score: ', total_f1)
    print('Plots saved to:', plot_dir)
    print('='*20)
    
    plotter = Plotter(class_num=2, plots_dir=plot_dir)
    plotter = Plotter.cnf_matrix(file_name="")

    plotter.roc_curve(f'roc_{model_name}.png', all_labels, all_probs)
    plotter.prc_curve(f'prc_{model_name}.png', all_labels, all_probs)
    plotter.cnf_matrix(f'cnf_{model_name}.png', all_labels, all_predictions)
    
    ClassificationReport(file_path=plot_dir.joinpath(f'classification_report_{model_name}.txt'),
                         pred=all_predictions,
                         target=all_labels)

def test_function(config_dict: dict,
                  model):
    test_dataset = EdgeDataset(
        base_dir=config_dict['data_path_test'],
        batch_size=config_dict['batch_size'],
        shuffle=True,
        device=torch.device("cpu")
    )
    
    testLoader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=10,
        pin_memory=False
    )
    
    batch_x, batch_y = next(iter(testLoader))
    batch_x = batch_x.to(config_dict['device'])

    model.eval()
    outputs = model(batch_x)

    if outputs.shape == (batch_x.shape[0], config_dict['num_classes']):
        print('Model works as expected')
    else:
        print(f'Model does not work as expected\n')
        print(f'Expected output shape: (batch_x.shape[0], {config_dict["num_classes"]})\nReceived: {outputs.shape}')

def parser():
        
    """
    Parse command-line arguments for automated CNN training pipeline configuration.
    Accepts model naming, computational device selection (CPU/CUDA/GPU), and optional test mode activation.
    Returns parsed arguments with validation for device choices and formatted help text display.
    """
    
    parser = argparse.ArgumentParser(
        description="Script for testing the choosen model",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help=(
            "Base of the model's name.\n"
            "When iterating, name also gets an ID."
        )
    )

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
        choices=[0, 1],
        help=(
            "Device for tensor based computation.\n"
            'Pick:\n'
            '0: testing mode - check if model compiles and works as expected\n'
            '1: evaluate trained model'
        )
    )

    return parser.parse_args()

def main():
    args = parser()
    base_path = pth.Path(__file__).parent
    device_name = args.device
    device = torch.device('cuda') if (('cuda' in device_name.lower() or 'gpu' in device_name.lower()) and torch.cuda.is_available()) else torch.device('cpu')

    model_name = args.model_name
    model_name_no_num = model_name.rsplit('_', 1)[0]

    model_dir = base_path.joinpath(f'training_results/{model_name_no_num}')
    config_trained_dir = model_dir.joinpath('dict_files')
    model_path = config_trained_dir.joinpath(f'{model_name}_config.json')
    plot_dir = model_dir.joinpath('plots')

    config_dict = load_json(model_path)
    config_dict = convert_str_values(config_dict)
    config_dict['device'] = device


    model = AffinityMLP(config_dict['model_config'], scaling_config=config_dict['scaling_config'])
    model = load_model(file_path=model_dir.joinpath(f'{model_name}.pt'),
                       model=model,
                       device=device)
    model.eval()

    if args.mode == 0:
        test_function(config_dict, model)
    elif args.mode == 1:
        eval_model_front(config_dict=config_dict,
                         model=model,
                         paths=[model_path,
                                plot_dir])
        



if __name__ == '__main__':
    main()





