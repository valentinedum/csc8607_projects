"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

import argparse
import yaml
from utils import set_seed, get_device
from data_loading import get_dataloaders
from model import build_model
from train import calculate_loss_and_accuracy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from tqdm import tqdm
import gc
from torch.amp import autocast, GradScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--comparison_training", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device(prefer="auto")
    criterion = nn.CrossEntropyLoss()

    # Lire les grilles d'hyperparamètres
    if args.comparison_training is None:
        print("Grid Search Mode")
        num_epochs = 3
        hparams_config = config["hparams"]
        hparams_combinations = list(product(
            hparams_config['lr'],
            hparams_config['weight_decay'],
            hparams_config['num_blocks'],
            hparams_config['groups']
        ))
    else:
        print("Comparison Mode Training")
        num_epochs = 10
        hparams_combinations = [
            (run['lr'], run['weight_decay'], run['num_blocks'], run['groups'], run['comparison'])
            for run in config["comparison_hparams"]['runs']
        ]
    
    print(f"Nombre de combinaisons à tester : {len(hparams_combinations)}")

    # --- BOUCLE PRINCIPALE SUR LES COMBINAISONS ---
    for idx, combo in enumerate(hparams_combinations):
        if args.comparison_training is None:
            lr, weight_decay, num_blocks, groups = combo
            runs_dir = config['paths']['runs_dir'] + "/grid_search"
        else:
            lr, weight_decay, num_blocks, groups, comparison = combo
            runs_dir = config['paths']['runs_dir'] + f"/comparison_training/{comparison}"

        run_name = f"proj22_lr={lr}_wd={weight_decay}_blk={num_blocks}_grp={groups}"  
        print(f"\n--- Run {idx+1}/{len(hparams_combinations)} ---")
        print(f"LR={lr}, WD={weight_decay}, Blocks={num_blocks}, Groups={groups}")
        
        set_seed(config['train']['seed'])
        
        # Mettre à jour la config
        config['train']['optimizer']['lr'] = lr
        config['train']['optimizer']['weight_decay'] = weight_decay
        config['model']['num_blocks'] = num_blocks
        config['model']['groups'] = groups
        
        # Charger les données
        train_loader, val_loader, _, _ = get_dataloaders(config)
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialiser TensorBoard Writer pour ce run
        writer = SummaryWriter(log_dir=f"{runs_dir}/{run_name}")

        # Initialiser les trackers
        best_val_acc = 0.0
        final_val_loss = 0.0
        scaler = GradScaler(device=device)
        accumulation_steps = 2

        # --- ENTRAÎNEMENT ---
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                with autocast(device_type=device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
            
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * accumulation_steps  
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct_train / total_train  

            # Validation
            val_loss, val_accuracy = calculate_loss_and_accuracy(val_loader, model, criterion, device)
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            final_val_loss = val_loss

            # Log les courbes (pour SCALARS)
            writer.add_scalar('train/loss', avg_train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('train/accuracy', train_accuracy, epoch)
            writer.add_scalar('val/accuracy', val_accuracy, epoch)
    

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")

        # --- HPARAMS ---
        hparams_dict = {
            'lr': lr,
            'weight_decay': weight_decay,
            'num_blocks': num_blocks,
            'groups': groups
        }
        
        metric_dict = {
            'val/accuracy': best_val_acc,
            'val/loss': final_val_loss
        }
        
        writer.flush()
        writer.add_hparams(hparams_dict, metric_dict)
        writer.flush()
        writer.close()

        # --- NETTOYAGE MÉMOIRE ---
        del model, optimizer, train_loader, val_loader, writer, scaler
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print("\nGrid Search terminée !")

if __name__ == "__main__":
    main()
