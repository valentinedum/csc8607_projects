"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""
import torch
from torch.utils.data import DataLoader
from pandas import read_parquet
import os
from datasets import load_dataset, ClassLabel, Image
from torchvision import transforms as T
from augmentation import get_augmentation_transforms
from preporcessing import get_preprocess_transforms

def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    À implémenter.
    """

    # Chargement des données
    root_path = os.path.expanduser(config['dataset']['root'])
    
    data_files = {
        "train": os.path.join(root_path, config['dataset']['split']['train'], "*.parquet"),
        "test": os.path.join(root_path, config['dataset']['split']['test'], "*.parquet")
    }

    dataset = load_dataset('parquet', data_files=data_files)
    columns_to_keep = ['image', 'label']
    dataset = dataset.remove_columns([col for col in dataset['train'].column_names if col not in columns_to_keep])
    
    # Convertir la colonne label en ClassLabel pour stratifier correctement
    num_classes = config['model']['num_classes']
    class_label_feature = ClassLabel(num_classes=num_classes, names=[str(i) for i in range(num_classes)])
    dataset = dataset.cast_column('label', class_label_feature)
    dataset = dataset.cast_column('image', Image())

    trainset = dataset['train']
    testset = dataset['test']

    # Création du val set à partir du train set
    split = trainset.train_test_split(
        test_size=0.2,
        stratify_by_column="label", 
        seed=config["train"]["seed"]
    )

    trainset = split['train']
    valset = split['test']

    # Appliquer les transformations
    preprocess_transforms = get_preprocess_transforms(config)
    augment_transforms = get_augmentation_transforms(config)

    train_pipeline = T.Compose([
        augment_transforms,
        preprocess_transforms,
    ])

    val_test_pipeline = preprocess_transforms
    
    trainset.set_transform(lambda batch: {
        'image': [train_pipeline(img) for img in batch['image']],
        'label': batch['label']
    })
    valset.set_transform(lambda batch: {
        'image': [val_test_pipeline(img) for img in batch['image']],
        'label': batch['label']
    })
    testset.set_transform(lambda batch: {
        'image': [val_test_pipeline(img) for img in batch['image']],
        'label': batch['label']
    })

    meta = {
        "num_classes": config['model']['num_classes'],
        "input_shape": tuple(config['model']['input_shape']), 
    }

    train_loader = DataLoader(trainset, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(valset, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(testset, batch_size=config['train']['batch_size'], shuffle=False)


    return train_loader, val_loader, test_loader, meta
    # raise NotImplementedError("get_dataloaders doit être implémentée par l'étudiant·e.")

## Test rapide
if __name__ == "__main__":
    
    import yaml
    from torch.utils.data import RandomSampler, SequentialSampler
    import sys

    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("--- Chargement des données... ---")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader:   {len(val_loader)} batches")
    print(f"Test loader:  {len(test_loader)} batches")
    print(f"Meta:         {meta}\n")

    print("--- Vérification des Shapes (Batchs) ---")
    # Récupère le premier batch de chaque loader
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    train_images, train_labels = train_batch['image'], train_batch['label']
    val_images, val_labels = val_batch['image'], val_batch['label']
    test_images, test_labels = test_batch['image'], test_batch['label']

     # Affiche les shapes
    print(f"Shape du batch d'images (Train): {train_images.shape}")
    print(f"Shape du batch de labels (Train): {train_labels.shape}\n")
    print(f"Shape du batch d'images (Val):   {val_images.shape}")
    print(f"Shape du batch de labels (Val):   {val_labels.shape}\n")
    print(f"Shape du batch d'images (Test):  {test_images.shape}")
    print(f"Shape du batch de labels (Test):  {test_labels.shape}\n")

