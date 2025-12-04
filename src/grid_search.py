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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # À implémenter par l'étudiant·e :
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    hparams_config = config["hparams"]
    # raise NotImplementedError("grid_search.main doit être implémenté par l'étudiant·e.")

if __name__ == "__main__":
    main()