"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse
import math
from data_loading import get_dataloaders
from model import build_model
from utils import set_seed, get_device, count_parameters
import yaml
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # À implémenter par l'étudiant·e :

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['train']['seed'])
    
    device = get_device(prefer="auto")
    print("Device : ", device)

    start = 1e-7
    end = 1e-1
    num_steps = 100
    weight_decay = float(config['train']['optimizer']['weight_decay'])
    learning_rates = np.logspace(math.log10(start), math.log10(end), num=num_steps)

    train_loader, _, _, _ = get_dataloaders(config)
    model = build_model(config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start, weight_decay=weight_decay)
    
    runs_dir = config['paths']['runs_dir']
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir="runs/lr_finder")
    
    best_loss = float('inf')
    avg_loss = 0.0
    losses_log = []
    lrs_log = []

    iter_loader = iter(train_loader)
    
    print(f"--- Démarrage LR Finder ({num_steps} itérations) ---")
    pbar = tqdm(enumerate(learning_rates), total=len(learning_rates), desc="LR Finder")

    for step, lr in pbar:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        try :
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)
          
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if not torch.isfinite(loss):
            print("Stopping early, loss is non-finite")
            break

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if step == 0:
            avg_loss = current_loss
        else:
            avg_loss = 0.98 * avg_loss + 0.02 * current_loss

        if avg_loss < best_loss:
            best_loss = avg_loss
        
        losses_log.append(avg_loss)
        lrs_log.append(lr)
        
        if avg_loss > 4 * best_loss and step > 10:
            print("Stopping early, loss has diverged")
            break


        writer.add_scalar("lr_finder/loss", avg_loss, step)
        writer.add_scalar("lr_finder/lr", lr, step)

        pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'lr': f"{lr:.1e}"})

    writer.close()
    print(f"Meilleur lr : {lrs_log[np.argmin(losses_log)]} avec loss : {min(losses_log):.4f}")

    # raise NotImplementedError("lr_finder.main doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    main()