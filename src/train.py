"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""
import math
from data_loading import get_dataloaders
from model import build_model
from utils import set_seed, get_device, count_parameters
import argparse
import yaml
from tqdm import tqdm
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18  # Pour test rapide (à supprimer ensuite)
import torch.nn as nn

def calculate_loss_and_accuracy(loader, model, criterion, device):
    """Calcule la loss et l'accuracy sur un loader complet."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # Gestion du format Dict retourné par Hugging Face
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    # Ajoutez d'autres arguments si nécessaire (batch_size, lr, etc.)
    args = parser.parse_args()
    # À implémenter par l'étudiant·e :

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)
    
    device = get_device(prefer="auto")
    print("Device : ", device)

    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    model = build_model(config).to(device)
    # model = resnet18(weights=None) 
    # model.fc = nn.Linear(512, 200) # On adapte la fin pour 200 classes
    # model = model.to(device)

    learning_rate = config['train']['optimizer']['lr']
    weight_decay = 0.0
    num_epochs = args.max_epochs or config['train']['epochs']
    print(f"Nombre d'epochs : {num_epochs}")
    batch_size = config['train']['batch_size']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    runs_dir = config['paths']['runs_dir']
    os.makedirs(runs_dir, exist_ok=True)

    # --- PREMIER BATCH ---
    writer_batch = SummaryWriter(log_dir=os.path.join(runs_dir, "first_batch"))
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    print(f"\nPremier batch - forme des images : {images.shape}, forme des labels : {labels.shape}")
    writer_batch.add_text("Batch Info",f"Forme des images : {images.shape}, Forme des labels : {labels.shape}")

    model.train()
    outputs = model(images)
    print(f"Forme de la sortie du modèle sur le premier batch : {outputs.shape}\n")
    writer_batch.add_text("Output Info", f"Forme de la sortie : {outputs.shape}")

    initial_loss = criterion(outputs, labels)
    loss_value = initial_loss.item()
    num_classes = config['model']['num_classes']
    theoretical_loss = - math.log(1 / num_classes)

    print(f"\nLoss Initiale calculée : {loss_value:.4f}")
    writer_batch.add_scalar('Loss/Initial', loss_value, 0)
    print(f"Loss Théorique (Uniforme sur {num_classes} classes) : {theoretical_loss:.4f}\n")
    writer_batch.add_scalar('Loss/Theoretical', theoretical_loss, 0)

    optimizer.zero_grad()
    initial_loss.backward()
    writer_batch.close()

    # --- OVERFIT SMALL ---
    if args.overfit_small:
        print("!!! MODE OVERFIT SMALL ACTIVÉ !!!")
        writer_overfit = SummaryWriter(log_dir=os.path.join(runs_dir, "overfit_small"))
        images = images[:16]
        labels = labels[:16]
        

        pbar = tqdm(range(50), desc="Overfitting")
        
        for i in pbar:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # On logge dans 'Train/Loss' pour voir la courbe descendre
            writer_overfit.add_scalar('Train/Loss', loss.item(), i)
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        print("\nFin du test Overfit. Vérifiez TensorBoard.")
        print("Loss finale sur le petit échantillon : {:.6f}".format(loss.item()))
        writer_overfit.close()
        return

    # --- ENTRAÎNEMENT COMPLET ---
    else:
        writer = SummaryWriter(log_dir=os.path.join(runs_dir, "training"))
        weight_decay = config['train']['optimizer']['weight_decay']
        
        print("ENTRAÎNEMENT COMPLET") 
        print(f"LR={learning_rate}, WD={weight_decay}, num_blocks={config['model']['num_blocks']}, groups={config['model']['groups']}")
        print(f"Batch size={batch_size}, Epochs={num_epochs}\n")
        
        best_val_acc = 0.0
        best_checkpoint_path = os.path.join(config['paths']['artifacts_dir'], 'best.ckpt')
        os.makedirs(config['paths']['artifacts_dir'], exist_ok=True)

        for epoch in range(num_epochs):
            # --- PHASE TRAIN ---
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]", leave=True)
            
            for batch in progress_bar:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct_train / total_train

            # --- PHASE VALIDATION ---
            val_loss, val_accuracy = calculate_loss_and_accuracy(val_loader, model, criterion, device)
            
            # Sauvegarder le meilleur checkpoint
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print(f"Nouveau meilleur checkpoint sauvegardé (acc={val_accuracy:.4f})")

            # Logger les métriques
            writer.add_scalar('train/loss', avg_train_loss, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('train/accuracy', train_accuracy, epoch)
            writer.add_scalar('val/accuracy', val_accuracy, epoch)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_accuracy:.4f}")

        print(f"\nEntraînement terminé!")
        print(f"Meilleur checkpoint: {best_checkpoint_path} (val_acc={best_val_acc:.4f})")
        
        writer.close()

if __name__ == "__main__":
    main()