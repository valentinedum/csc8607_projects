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

def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python). À implémenter."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # raise NotImplementedError("set_seed doit être implémentée par l'étudiant·e.")


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto'). À implémenter."""
    raise NotImplementedError("get_device doit être implémentée par l'étudiant·e.")


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle. À implémenter."""
    raise NotImplementedError("count_parameters doit être implémentée par l'étudiant·e.")


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir. À implémenter."""
    raise NotImplementedError("save_config_snapshot doit être implémentée par l'étudiant·e.")