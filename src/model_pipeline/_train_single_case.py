import pathlib as pth
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Generator, Tuple

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from AffinityMLP import AffinityMLP
from _data_loader import EdgeDataset

from utils import get_dataset_len
from utils.weights import calculate_binary_weights
from utils.metrics import binary_f1_score


def train_model(
    config: Dict[str, Any]
) -> Generator[Tuple[AffinityMLP, Dict[str, list]], None, None]:
    
    train_dataset = EdgeDataset(
        base_dir=config['data_path_train'],
        batch_size=config['batch_size'],
        shuffle=True,
        device=torch.device('cpu')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=10,
        pin_memory=False
    )

    val_dataset = EdgeDataset(
        base_dir=config['data_path_val'],
        batch_size=config['batch_size'],
        shuffle=False,
        device=torch.device('cpu')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=10,
        pin_memory=False
    )

    total_t = get_dataset_len(train_loader)
    total_v = get_dataset_len(val_loader)

    assert total_t > 0, "Training dataset is empty."
    assert total_v > 0, "Validation dataset is empty."
    
    weights_t = calculate_binary_weights(train_loader, total=total_t, verbose=False)
    weights_v = calculate_binary_weights(val_loader, total=total_v, verbose=False)

    model = AffinityMLP(config['model_config'])
    model.to(config['device'])
    
    weights_t = torch.tensor(weights_t, dtype=torch.float32)
    weights_v = torch.tensor(weights_v, dtype=torch.float32)

    criterion_t = nn.BCEWithLogitsLoss(pos_weight=weights_t).to(config['device'])
    criterion_v = nn.BCEWithLogitsLoss(pos_weight=weights_v).to(config['device'])
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_t*config['epochs'],
        pct_start=config['pct_start'],
        anneal_strategy='cos',
        div_factor=config['div_factor'],
        final_div_factor=config['final_div_factor']
    )
    
    loss_hist, f1_hist = [], []
    loss_v_hist, f1_v_hist = [], []
    
    for epoch in tqdm(range(config['epochs']), desc="Epochs"):
        
        # Training
        model.train()
        epoch_loss, epoch_f1, epoch_samples = 0.0, 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{config['epochs']}", total=total_t, leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(config['device'])
            batch_y = batch_y.to(config['device'])
            
            logits = model(batch_x)
            loss = criterion_t(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            preds = torch.sigmoid(logits)
            
            batch_f1 = binary_f1_score(preds, batch_y)
            
            epoch_loss += loss.item() * batch_y.size(0)
            epoch_f1 += batch_f1 * batch_y.size(0)
            epoch_samples += batch_y.size(0)
            
            pbar.set_postfix({
                'loss': f"{epoch_loss/epoch_samples:.4f}",
                'f1': f"{epoch_f1/epoch_samples:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        loss_hist.append(epoch_loss / epoch_samples)
        f1_hist.append(epoch_f1 / epoch_samples)
        
        # Validation
        model.eval()
        epoch_loss_v, epoch_f1_v, epoch_samples_v = 0.0, 0.0, 0
        
        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"Val {epoch+1}/{config['epochs']}", total=total_v, leave=False)
            for batch_x, batch_y in pbar_v:
                batch_x = batch_x.to(config['device'])
                batch_y = batch_y.to(config['device'])
                
                logits = model(batch_x)
                loss = criterion_v(logits, batch_y)
                
                preds = torch.sigmoid(logits)
                
                batch_f1_v = binary_f1_score(preds, batch_y)
                
                epoch_loss_v += loss.item() * batch_y.size(0)
                epoch_f1_v += batch_f1_v * batch_y.size(0)
                epoch_samples_v += batch_y.size(0)
                
                pbar_v.set_postfix({
                    'loss': f"{epoch_loss_v/epoch_samples_v:.4f}",
                    'f1': f"{epoch_f1_v/epoch_samples_v:.4f}"
                })
        
        loss_v_hist.append(epoch_loss_v / epoch_samples_v)
        f1_v_hist.append(epoch_f1_v / epoch_samples_v)
        
        hist_dict = {
            'loss_hist': loss_hist,
            'f1_hist': f1_hist,
            'loss_v_hist': loss_v_hist,
            'f1_v_hist': f1_v_hist
        }
        
        yield model, hist_dict