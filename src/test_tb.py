import torch
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time

# 1. Nettoyage violent (On efface tout avant de commencer)
if os.path.exists("runs_test"):
    shutil.rmtree("runs_test")

# 2. Simulation de 3 runs
for i, lr in enumerate([0.1, 0.01, 0.001]):
    print(f"Test Run {i} avec LR={lr}")
    
    # Structure de dossier claire
    writer = SummaryWriter(f"runs_test/experiment_{i}")
    
    # Simulation d'entraînement
    for step in range(10):
        # On invente une accuracy qui monte
        fake_acc = 0.5 + (i * 0.1) + (step * 0.01)
        writer.add_scalar("val/accuracy", fake_acc, step)
    
    # 3. Écriture des HParams
    # On lie explicitement la métrique HParam à la DERNIÈRE valeur de la courbe
    hparams = {"lr": lr, "bs": 32}
    metrics = {"val/accuracy": fake_acc} # Doit matcher le nom ci-dessus
    
    writer.add_hparams(hparams, metrics)
    
    writer.flush()
    writer.close()
