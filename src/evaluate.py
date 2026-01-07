"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from model import build_model
from data_loading import get_dataloaders
from utils import get_device, set_seed


def evaluate(model, test_loader, criterion, device):
    """Évalue le modèle sur le test set."""
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Charger config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device(prefer="auto")
    
    # Charger données, modèle et checkpoint
    _, _, test_loader, meta = get_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Test set: {len(test_loader.dataset)} exemples\n")
    
    # Évaluer
    results = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    report = classification_report(results['labels'], results['predictions'], zero_division=0)
    cm = confusion_matrix(results['labels'], results['predictions'])
    print("--- Résultats d'Évaluation sur le Test Set ---")
    print(f"\nLoss: {results['loss']:.4f}")
    print(f"\nAccuracy: {results['accuracy']:.2f}%")
    print(f"\nClassification Report: \n{report}")
    print(f"\nConfusion Matrix (extrait 10x10) : {cm}")
    
    # Logger dans TensorBoard
    writer = SummaryWriter(log_dir=config['paths']['runs_dir'] + "/evaluation")
    writer.add_scalar('test/loss', results['loss'], 0)
    writer.add_scalar('test/accuracy', results['accuracy'], 0)
    writer.add_scalar('val/accuracy', checkpoint.get('val_accuracy', 0.0) * 100, 0)
    writer.add_text('Classification Report', report.replace('\n', '  \n'), 0)
    
    # Ajouter la confusion matrix comme image
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_title(f'Confusion Matrix ({meta["num_classes"]} classes)')
    ax.set_xlabel('Classe prédite')
    ax.set_ylabel('Vraie classe')
    plt.colorbar(im, ax=ax)
    writer.add_figure('Confusion Matrix', fig, 0)
    plt.close(fig)
    
    writer.close()


if __name__ == "__main__":
    main()