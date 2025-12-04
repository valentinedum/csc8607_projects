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
    
    device = get_device(prefer="cpu")
    print("Device : ", device)

    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    model = build_model(config).to(device)

    learning_rate = config['train']['optimizer']['lr']
    weight_decay = 0.0
    num_epochs = args.max_epochs or config['train']['epochs']
    batch_size = config['train']['batch_size']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    runs_dir = config['paths']['runs_dir']
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_dir)

    # Premier batch
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    print(f"\nPremier batch - forme des images : {images.shape}, forme des labels : {labels.shape}")
    writer.add_text("Batch Info",f"Forme des images : {images.shape}, Forme des labels : {labels.shape}")

    model.train()
    outputs = model(images)
    print(f"Forme de la sortie du modèle sur le premier batch : {outputs.shape}\n")
    writer.add_text("Output Info", f"Forme de la sortie : {outputs.shape}")

    initial_loss = criterion(outputs, labels)
    loss_value = initial_loss.item()
    num_classes = config['model']['num_classes']
    theoretical_loss = - math.log(1 / num_classes)

    print(f"\nLoss Initiale calculée : {loss_value:.4f}")
    writer.add_scalar('Loss/Initial', loss_value, 0)
    print(f"Loss Théorique (Uniforme sur {num_classes} classes) : {theoretical_loss:.4f}\n")
    writer.add_scalar('Loss/Theoretical', theoretical_loss, 0)

    optimizer.zero_grad()
    initial_loss.backward()

    # grad_norm = 0.0
    # for param in model.parameters():
    #     if param.grad is not None:
    #         grad_norm += param.grad.data.norm(2).item() ** 2
    # grad_norm = grad_norm ** 0.5
    # print(f"Gradient norm on first batch: {grad_norm:.4f}")
    # writer.add_scalar('GradNorm/Initial', grad_norm, 0)

    if args.overfit_small:
        print("!!! MODE OVERFIT SMALL ACTIVÉ !!!")
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
            writer.add_scalar('Train/Loss', loss.item(), i)
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        print("\nFin du test Overfit. Vérifiez TensorBoard.")
        writer.close()
        return


    # for epoch in range(num_epochs):
    #     model.train()
    #     train_loss = 0.0
    #     correct_train = 0
    #     total_train = 0

    #     for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad(set_to_none=True)
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         train_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         correct_train += (predicted == labels).sum().item()
    #         total_train += labels.size(0)

    #     train_loss = train_loss / len(train_loader)
    #     train_accuracy = correct_train / total_train

    #     val_accuracy, val_loss = calculate_loss_and_accuracy(val_loader, model, criterion, device)

    #     writer.add_scalar('Loss/Train', train_loss, epoch)
    #     writer.add_scalar('Loss/Validation', val_loss, epoch)
    #     writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    #     writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

    #     print(
    #         f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # # Final test loss and accuracy
    # test_loss, test_accuracy = calculate_loss_and_accuracy(test_loader, model, criterion)
    # writer.add_scalar('Loss/Test', test_loss, num_epochs)
    # writer.add_scalar('Accuracy/Test', test_accuracy, num_epochs)
    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    writer.close()

    # raise NotImplementedError("train.main doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    main()