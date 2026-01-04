"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from torchvision import transforms as T

def get_preprocess_transforms(config: dict, apply_random_erasing=False):
    """Retourne les transformations de pré-traitement. À implémenter."""
    input_shape = config['model']['input_shape']
    target_size = (input_shape[1], input_shape[2]) # (224, 224)

    norm_config = config['preprocess']['normalize']
    mean = norm_config['mean']
    std = norm_config['std']

    transform_list = [
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    
    # RandomErasing (après ToTensor et Normalize)
    if apply_random_erasing:
        random_erasing_params = config['augment'].get('random_erasing')
        if random_erasing_params:
            transform_list.append(T.RandomErasing(
                p=random_erasing_params.get('p', 0.1),
                scale=tuple(random_erasing_params.get('scale', [0.02, 0.08])),
                value=random_erasing_params.get('value', 'random')
            ))
    
    preprocess = T.Compose(transform_list)
    return preprocess
    # raise NotImplementedError("get_preprocess_transforms doit être implémentée par l'étudiant·e.")