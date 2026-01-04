"""
Data analysis (Train, Val, Test)
"""
import yaml
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from torchvision import transforms as T
from random import sample as randomsample
from datasets import load_dataset, ClassLabel, Image
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from preporcessing import get_preprocess_transforms
from augmentation import get_augmentation_transforms
from model import build_model
from utils import count_parameters

def get_sorted_counts(dataset, num_classes):
    """
    Retourne une liste de nombres d'exemples par classe.
    """
    labels = dataset['label'] 
    counts = Counter(labels)
    return [counts.get(i, 0) for i in range(num_classes)]

def plot_distribution(sorted_counts, class_names, title):
    """
    Crée un graphique de la distribution des classes.
    """
    width = max(12, len(class_names) / 5)
    fig, ax = plt.subplots(figsize=(width, 5))
    
    indices = range(len(class_names))
    ax.bar(indices, sorted_counts)
    
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'exemples")

    step = max(1, len(class_names) // 30)
    ax.set_xticks(list(indices)[::step])
    ax.set_xticklabels([class_names[i] for i in indices][::step], rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    return fig

def compute_statistics(dataset, config):
    """
    Calcule moyenne/std sur un echantillon.
    """
    print("   Calcul des stats (sur un échantillon de 1000 images)...")
    target_size = tuple(config['model']['input_shape'][1:]) # (224, 224)
    
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor()
    ])
    
    # Echantillonnage
    indices = randomsample(range(len(dataset)), min(len(dataset), 1000))
    tensor_images = [transform(dataset[i]['image']) for i in indices]
    
    tensor_images = torch.stack(tensor_images)
    mean = tensor_images.mean(dim=(0, 2, 3))
    std = tensor_images.std(dim=(0, 2, 3))
    return mean, std

def missing_values(dataset, feature):
    """
    Compte le nombre de valeurs manquantes (None) pour une colonne donnée.
    """
    count = 0
    for item in dataset:
        if item[feature] is None:
            count += 1
    return count

def image_analysis(dataset):
    """
    Analyse les tailles et modes. Sur échantillon aussi.
    """
    print("   Analyse des tailles (sur un échantillon de 1000 images)...")
    tailles = Counter()
    modes = Counter()
    
    indices = randomsample(range(len(dataset)), min(len(dataset), 1000))
    
    for i in indices:
        img = dataset[i]['image'] 
        tailles[img.size] += 1
        modes[img.mode] += 1

    return tailles.most_common(5), modes.most_common(5)

def calculate_baselines(trainset, testset, num_classes, config):
    """Calcule des baselines simples (Classe majoritaire et aléatoire)."""
    train_labels = trainset['label']
    test_labels = testset['label']
    
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro') # Micro = Global accuracy
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro')

    n_samples = len(test_labels)
    torch.manual_seed(config["train"]["seed"])

    # 1. Classe majoritaire
    majority_class_label = Counter(train_labels).most_common(1)[0][0]
    preds_major_indices = torch.full((n_samples,), majority_class_label, dtype=torch.long)

    accuracy = acc_metric(preds_major_indices, torch.tensor(test_labels)).item()
    f1 = f1_metric(preds_major_indices, torch.tensor(test_labels)).item()
    print(f"Baseline Classe Majoritaire -> Classe: {majority_class_label}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    # 2. Aléatoire
    random_class = Counter(train_labels).elements()
    scores_random = torch.rand(n_samples, num_classes)
    preds_random_indices = torch.argmax(scores_random, dim=1)

    accuracy = acc_metric(preds_random_indices, torch.tensor(test_labels)).item()
    f1 = f1_metric(preds_random_indices, torch.tensor(test_labels)).item() 
    print(f"Baseline Aléatoire -> Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

def analyze_data():
    """Analyse les données et log les distributions dans TensorBoard."""
    
    print("--- Démarrage de l'analyse ---")
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. CHARGEMENT MANUEL DES DONNÉES
    root_path = os.path.expanduser(config['dataset']['root'])
    data_files = {
        "train": os.path.join(root_path, config['dataset']['split']['train'], "*.parquet"),
        "test": os.path.join(root_path, config['dataset']['split']['test'], "*.parquet")
    }
    
    print(f"Chargement depuis : {root_path}")
    dataset = load_dataset('parquet', data_files=data_files)
    
    # Nettoyage colonnes
    columns_to_keep = ['image', 'label']
    dataset = dataset.remove_columns([c for c in dataset['train'].column_names if c not in columns_to_keep])
    
    # Casting
    num_classes = config['model']['num_classes']
    class_label_feature = ClassLabel(num_classes=num_classes, names=[str(i) for i in range(num_classes)])
    dataset = dataset.cast_column('label', class_label_feature)
    dataset = dataset.cast_column('image', Image()) # Décode en PIL

    # Split (Même logique que data_loading.py)
    trainset_full = dataset['train']
    testset = dataset['test']
    
    split = trainset_full.train_test_split(
        test_size=0.2, 
        stratify_by_column="label", 
        seed=config["train"]["seed"]
    )
    
    train = split['train']
    val = split['test']
    test = testset
    
    class_names = train.features['label'].names


    # 2. STATISTIQUES DE DISTRIBUTION
    print("Génération des graphiques de distribution...")
    writer = SummaryWriter(log_dir="runs/data_analysis")
    train_counts = get_sorted_counts(train, num_classes)
    val_counts = get_sorted_counts(val, num_classes)
    test_counts = get_sorted_counts(test, num_classes)

    fig_train = plot_distribution(train_counts, class_names, "Distribution Train")
    fig_val = plot_distribution(val_counts, class_names, "Distribution Val")
    fig_test = plot_distribution(test_counts, class_names, "Distribution Test")

    writer.add_figure("Distribution/Train", fig_train)
    writer.add_figure("Distribution/Val", fig_val)
    writer.add_figure("Distribution/Test", fig_test)
    
    print(f"Tailles -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # 3. ANALYSE DES VALEURS MANQUANTES
    for feature in train.features:
        train_missing = missing_values(train, feature)
        val_missing = missing_values(val, feature)
        test_missing = missing_values(test, feature)
        print(f"Valeurs manquantes pour '{feature}' -> Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")
    
    # 4. ANALYSE DES IMAGES (Tailles & Modes)
    train_tailles, train_modes = image_analysis(train)
    val_tailles, val_modes = image_analysis(val)
    test_tailles, test_modes = image_analysis(test)
    print(f"Nombre de tailles d'images différentes - Train: {len(train_tailles)}, Val: {len(val_tailles)}, Test: {len(test_tailles)}")
    print(f"Modes d'images - Train: {train_modes}, Val: {val_modes}, Test: {test_modes}")
    
    # 5. STATISTIQUES DE PIXELS (Mean/Std)
    mean, std = compute_statistics(train, config)
    print(f"Stats calculées -> Mean: {mean.numpy()}, Std: {std.numpy()}")
    
    # 6. VISUALISATION AUGMENTATION ET PREPROCESSING
    print("Génération des exemples d'augmentation...")
    
    augment_pipeline = get_augmentation_transforms(config)
    preprocess_pipeline = get_preprocess_transforms(config)
    
    indices = randomsample(range(len(train)), 5)
    raw_images = [train[i]['image'] for i in indices]

    # On va afficher : Originale -> Augmentée -> Finale
    for idx, pil_img in enumerate(raw_images):
        writer.add_image(f"Exemple_{idx}/Original", T.ToTensor()(pil_img), 0)
        
        aug_img = augment_pipeline(pil_img)
        writer.add_image(f"Exemple_{idx}/Augmented", T.ToTensor()(aug_img), 0)

        final_tensor = preprocess_pipeline(aug_img)
        
        writer.add_image(f"Exemple_{idx}/FinalInput", final_tensor, 0)

    writer.close()
    print("\nAnalyse terminée. Lancez TensorBoard : tensorboard --logdir runs/data_analysis")

    # 7. CALCUL DES BASELINES
    print("\nCalcul des baselines...")
    calculate_baselines(train, test, num_classes, config)

    # 8. PARAMETRES ENTRAINABLES
    countable_parameters = count_parameters(build_model(config))
    print(f"Nombre de paramètres entraînables : {countable_parameters}")

if __name__ == "__main__":
    analyze_data()