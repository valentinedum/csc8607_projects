"""
Data analysis (Train, Val, Test)
"""

import yaml
import src.data_loading as data_loading
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

def get_sorted_counts(dataset, num_classes):
    """
    Retourne une liste des comptes d'exemples par classe, triée par classe.
    """
    labels = [example['label'] for example in dataset]
    counts = Counter(labels)
    return [counts.get(i, 0) for i in range(num_classes)]

def plot_distribution(sorted_counts, class_names, title):
    """
    Crée un graphique de la distribution des classes.
    Avec les classes sur l'axe x et le nombre d'exemples sur l'axe y.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(class_names)), sorted_counts, marker='o', linestyle='-')
    ax.set_title(title)
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre d'exemples")
    xtick_positions = list(range(0, len(class_names), 20))
    xtick_labels = [class_names[i] for i in xtick_positions]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=8)
    plt.tight_layout()
    return fig

def missing_values(dataset, feature):
    """
    Compte le nombre de valeurs manquantes pour une feature donnée.
    """
    return sum(1 for example in dataset if example[feature] is None)

def image_analysis(dataset):
    """
    Analyse les différentes tailles et modes des images dans le dataset.
    Retourne des statistiques sur l'héterogeneité des images.
    """
    tailles = []
    modes = []
    for img in dataset['image']:
        taille = img.size  # (width, height)
        if taille not in tailles:
            tailles.append(taille)
        mode = str(img.mode)
        if mode not in modes:
            modes.append(mode)

    tailles = [f"{w}x{h}" for w, h in tailles]
    return tailles, modes

def analyze_data():
    """Analyse les données et log les distributions dans TensorBoard."""

    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train, val, test, meta = data_loading.get_dataloaders(config)
    class_names = train.dataset.features['label'].names
    num_classes = len(class_names)

    # # Comptage des classes
    # train_counts = get_sorted_counts(train.dataset, num_classes)
    # val_counts = get_sorted_counts(val.dataset, num_classes)
    # test_counts = get_sorted_counts(test.dataset, num_classes)

    # # Création des graphiques
    # fig_train = plot_distribution(train_counts, class_names, "Distribution des classes (Train Set)")
    # fig_val = plot_distribution(val_counts, class_names, "Distribution des classes (Validation Set)")
    # fig_test = plot_distribution(test_counts, class_names, "Distribution des classes (Test Set)")

    # # Log dans TensorBoard
    # writer = SummaryWriter("runs/data_analysis")
    # writer.add_figure("Train Class Distribution", fig_train)
    # writer.add_figure("Validation Class Distribution", fig_val)
    # writer.add_figure("Test Class Distribution", fig_test)
    # print("Num Classes/Train", num_classes)
    # print("Num Classes/Val", len(val.dataset.features['label'].names))
    # print("Num Classes/Test", len(test.dataset.features['label'].names))
    # print("Dataset Size/Train", len(train.dataset))
    # print("Dataset Size/Val", len(val.dataset))
    # print("Dataset Size/Test", len(test.dataset))
    # writer.close()

    # print("Analyse des valeurs manquantes :")
    # for feature in train.dataset.features:
    #     missing_train = missing_values(train.dataset, feature)
    #     missing_val = missing_values(val.dataset, feature)
    #     missing_test = missing_values(test.dataset, feature)
    #     print(f" - Feature '{feature}': Train={missing_train}, Val={missing_val}, Test={missing_test}")

    print("\nAnalyse des images :")
    train_tailles, train_modes = image_analysis(train.dataset)
    val_tailles, val_modes = image_analysis(val.dataset)
    test_tailles, test_modes = image_analysis(test.dataset)
    print(f"Nombre de tailles d'images différentes - Train: {len(train_tailles)}, Val: {len(val_tailles)}, Test: {len(test_tailles)}")
    print(f"Modes d'images - Train: {train_modes}, Val: {val_modes}, Test: {test_modes}")
    
    print("\n✅ Analyse terminée. Graphiques loggés dans TensorBoard (onglet 'Images').")

analyze_data()
