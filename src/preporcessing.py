"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from torchvision import transforms as T

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement. À implémenter."""
    input_shape = config['model']['input_shape']
    target_size = (input_shape[1], input_shape[2]) # (224, 224)

    norm_config = config['preprocess']['normalize']
    mean = norm_config['mean']
    std = norm_config['std']

    preprocess = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return preprocess
    # raise NotImplementedError("get_preprocess_transforms doit être implémentée par l'étudiant·e.")