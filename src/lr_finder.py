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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # À implémenter par l'étudiant·e :

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)
    
    device = get_device(prefer="cpu")
    print("Device : ", device)


    # raise NotImplementedError("lr_finder.main doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    main()