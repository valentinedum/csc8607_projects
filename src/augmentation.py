"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""
from torchvision import transforms as T

def get_augmentation_transforms(config: dict):
    """Retourne les transformations d'augmentation. À implémenter."""
    augment_config = config['augment']

    transform_list = []
    # 1. Random Horizontal Flip
    if augment_config.get('random_flip', False):
        transform_list.append(T.RandomHorizontalFlip(0.5))

    # 2. Color Jitter
    color_jitter_params = augment_config['color_jitter']
    if color_jitter_params:
        transform_list.append(T.ColorJitter(**color_jitter_params))

    # 3. Random Rotation
    if augment_config.get('random_rotation', False):
        transform_list.append(T.RandomRotation(degrees=15))

    return T.Compose(transform_list)
    # raise NotImplementedError("get_augmentation_transforms doit être implémentée par l'étudiant·e.")