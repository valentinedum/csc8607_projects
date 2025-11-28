"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python). À implémenter."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    writer = SummaryWriter(log_dir="runs/parameters")
    writer.add_scalar("Seed", seed)
    writer.close()
    # raise NotImplementedError("set_seed doit être implémentée par l'étudiant·e.")


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto'). À implémenter."""
    if prefer not in ["auto", None]:
        return prefer
    
    if torch.cuda.is_available():
        return "cuda"
    
    return "cpu"
    # raise NotImplementedError("get_device doit être implémentée par l'étudiant·e.")


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle. À implémenter."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # raise NotImplementedError("count_parameters doit être implémentée par l'étudiant·e.")


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir. À implémenter."""
    pass
    # raise NotImplementedError("save_config_snapshot doit être implémentée par l'étudiant·e.")