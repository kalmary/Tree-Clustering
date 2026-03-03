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

from EdgeGNN import EdgeClassifierGNN
from _data_loader import BatchedGraphDataset

from utils import get_dataset_len
from utils.weights import calculate_binary_weights
from utils.metrics import binary_f1_score, FocalLossBCE, ContrastiveLoss


def train_model(
    config: Dict[str, Any]
) -> Generator[Tuple[EdgeClassifierGNN, Dict[str, list]], None, None]:
    
    device_gpu = config['device']
    device_cpu = torch.device('cpu')

    device_loader = device_gpu
    device_loss = device_gpu
                                 
    
    train_dataset = BatchedGraphDataset(
        base_dir=config['data_path_train'],
        graphs_per_batch=config['batch_size'],
        shuffle=True,
        device=torch.device('cpu'),
        max_nodes=300,
        positive_bias=0.5
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=16,
        pin_memory=False
    )

    val_dataset = BatchedGraphDataset(
        base_dir=config['data_path_val'],
        graphs_per_batch=config['batch_size'],
        shuffle=False,
        device=torch.device('cpu'),
        max_nodes=300,
        positive_bias=0.0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=16,
        pin_memory=False
    )
    try:

        total_t = get_dataset_len(train_loader)
        total_v = get_dataset_len(val_loader)

        assert total_t > 0, "Training dataset is empty."
        assert total_v > 0, "Validation dataset is empty."
        
        weights_t = calculate_binary_weights(train_loader, total=total_t, verbose=False, return_pos_weight=True)
        weights_v = calculate_binary_weights(val_loader, total=total_v, verbose=False, return_pos_weight=True)

        model = EdgeClassifierGNN(config['model_config'], scaling_params=config['scaling_config'])
        model.to(config['device'])
        
        weights_t = torch.tensor(weights_t, dtype=torch.float32).to(device_loss)
        weights_v = torch.tensor(weights_v, dtype=torch.float32).to(device_loss)

        # criterion_t_focal = FocalLossBCE(pos_weight=weights_t, gamma = config['focal_gamma']).to(torch.device("cpu"))
        # criterion_v_focal = FocalLossBCE(pos_weight=weights_v, gamma = config['focal_gamma']).to(torch.device("cpu"))

        criterion_t_focal = nn.BCEWithLogitsLoss(pos_weight=weights_t).to(device_loss)
        criterion_v_focal = nn.BCEWithLogitsLoss(pos_weight=weights_v).to(device_loss)

        criterion_t_contrastive = ContrastiveLoss(alpha=weights_t, margin=0.15).to(device_loss)
        criterion_v_contrastive = ContrastiveLoss(alpha=weights_v, margin=0.15).to(device_loss)
        
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
            epoch_loss, epoch_loss_t_f, epoch_loss_t_c, epoch_f1, epoch_samples = 0.0, 0.0, 0.0, 0.0, 0
            
            pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{config['epochs']}", total=total_t, leave=False)
            for batch in pbar:
                batch = batch.to(config['device'])
                
                optimizer.zero_grad()
                logits, x, edge_index = model(batch, return_embeddings=True)

                # x = x.cpu()
                # edge_index = edge_index.cpu()

                focal_loss_t = criterion_t_focal(logits, batch.y.float())
                contrastive_loss_t = criterion_t_contrastive(x, edge_index, batch.y)

                loss = focal_loss_t + 0.25*contrastive_loss_t
                
                loss.backward()
                optimizer.step()
                
                try:
                    scheduler.step()
                except Exception:
                    pass

                logits = logits.cpu()
                batch = batch.cpu()
                
                preds = torch.sigmoid(logits.detach())
                
                batch_f1 = binary_f1_score(preds, batch.y.long())
                

                epoch_loss += loss.item() * batch.y.size(0)
                epoch_loss_t_c += contrastive_loss_t.item() * batch.y.size(0)
                epoch_loss_t_f += focal_loss_t.item() * batch.y.size(0)

                epoch_f1 += batch_f1 * batch.y.size(0)
                epoch_samples += batch.y.size(0)
                
                pbar.set_postfix({
                    'loss': f"{epoch_loss/epoch_samples:.4f}",
                    'loss_c': f"{epoch_loss_t_c/epoch_samples:.4f}",
                    'loss_f': f"{epoch_loss_t_f/epoch_samples:.4f}",
                    'f1': f"{epoch_f1/epoch_samples:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            
            loss_hist.append(epoch_loss / epoch_samples)
            f1_hist.append(epoch_f1 / epoch_samples)
            
            # Validation
            model.eval()
            epoch_loss_v, epoch_f1_v, epoch_loss_v_f, epoch_loss_v_c, epoch_samples_v = 0.0, 0.0, 0.0, 0.0, 0
            
            with torch.no_grad():
                pbar_v = tqdm(val_loader, desc=f"Val {epoch+1}/{config['epochs']}", total=total_v, leave=False)
                for batch in pbar_v:
                    batch = batch.to(config['device'])
                    
                    logits, x, edge_index = model(batch, return_embeddings = True)

                    # x = x.cpu()
                    # edge_index = edge_index.cpu()

                    focal_loss_v = criterion_v_focal(logits, batch.y.float())
                    contrastive_loss_v = criterion_v_contrastive(x, edge_index, batch.y)

                    loss = focal_loss_v + 0.25*contrastive_loss_v
                
                    logits = logits.cpu()
                    batch = batch.cpu()
                    
                    preds = torch.sigmoid(logits)
                    batch_f1_v = binary_f1_score(preds, batch.y.long())
                    
                    epoch_loss_v += loss.item() * batch.y.size(0)
                    epoch_loss_v_f += focal_loss_v.item() * batch.y.size(0)
                    epoch_loss_v_c += contrastive_loss_v.item() * batch.y.size(0)
                    epoch_f1_v += batch_f1_v * batch.y.size(0)
                    epoch_samples_v += batch.y.size(0)
                    
                    pbar_v.set_postfix({
                        'loss': f"{epoch_loss_v/epoch_samples_v:.4f}",
                        'loss_c': f"{epoch_loss_v_c/epoch_samples_v:.4f}",
                        'loss_f': f"{epoch_loss_v_f/epoch_samples_v:.4f}",
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

    except Exception as e:
        print(f"Error during training: {e}")
        try:
            del model
        except Exception as e:
            pass
        torch.cuda.empty_cache()
        yield None, {}
