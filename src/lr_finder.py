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
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    # À implémenter par l'étudiant·e :

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)
    
    device = get_device(prefer="auto")
    print("Device : ", device)

    start = 1e-7
    end = 1e-1
    num_steps = 50
    weight_decay = config['train']['optimizer']['weight_decay']
    learning_rates = np.logspace(math.log10(start), math.log10(end), num=num_steps)

    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    model = build_model(config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir="runs/lr_finder")
    
    losses = []
    best_loss = float('inf')
    best_lr = start

    pbar = tqdm(enumerate(learning_rates), total=len(learning_rates), desc="LR Finder")

    for step, lr in pbar:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 10:
                break 
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        if avg_loss > 4 * best_loss:
            print("Stopping early, loss has diverged")
            break
        
        if avg_loss < best_loss:
            best_loss = avg_loss

        writer.add_scalar("lr_finder/loss", avg_loss, step)
        writer.add_scalar("lr_Finder/lr", lr, step)

        print(f"Step {step+1}/{num_steps}, LR: {lr:.6f}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lr = lr

    
    print(f"Best LR: {best_lr:.6f} with Loss: {best_loss:.4f}")
    writer.close()

    # raise NotImplementedError("lr_finder.main doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    main()