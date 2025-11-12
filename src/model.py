"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn

def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config. À implémenter."""
    layers = []
    input_shape = config['model']['input_shape']
    num_classes = config['model']['num_classes']
    hidden_sizes = config['model']['hidden_sizes']
    activation_name = config['model']['activation']
    num_blocks = 2
    group = 2

    # Première couche d'initialisation
    layers.append(nn.Conv2d(in_channels=input_shape[0], out_channels=hidden_sizes[0], kernel_size=3, padding=1))
    layers.append(nn.BatchNorm2d(hidden_sizes[0]))
    if activation_name == 'relu':
        layers.append(nn.ReLU())
    
    # Stages 
    def create_block(in_channels, out_channels, activation_name, G=group, num_blocks=2):
        block_layers = []
        block_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        block_layers.append(nn.BatchNorm2d(out_channels))
        if activation_name == 'relu':
            block_layers.append(nn.ReLU())
        
        if out_channels % G != 0:
            raise ValueError(f"Le nombre de channels {out_channels} n'est pas divisible par le nombre de groupes {G}.")
        
        block_layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=G))
        block_layers.append(nn.BatchNorm2d(out_channels))
        if activation_name == 'relu':
            block_layers.append(nn.ReLU())
        
        return block_layers
    
    # Stage 1
    for i in range(num_blocks):
        layers.extend(create_block(hidden_sizes[0], hidden_sizes[0], activation_name))
    layers.append(nn.MaxPool2d(kernel_size=2))

    # Stage 2
    layers.extend(create_block(hidden_sizes[0], hidden_sizes[1], activation_name))
    for i in range(num_blocks - 1):
        layers.extend(create_block(hidden_sizes[1], hidden_sizes[1], activation_name))
    layers.append(nn.MaxPool2d(kernel_size=2))

    # Stage 3
    layers.extend(create_block(hidden_sizes[1], hidden_sizes[2], activation_name))
    for i in range(num_blocks - 1):
        layers.extend(create_block(hidden_sizes[2], hidden_sizes[2], activation_name))
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))

    layers.append(nn.Flatten())
    layers.append(nn.Linear(config['train']['batch_size'], num_classes))

    return nn.Sequential(*layers)


    # raise NotImplementedError("build_model doit être implémentée par l'étudiant·e.")