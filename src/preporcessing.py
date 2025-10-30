"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

def get_preprocess_transforms(config: dict):
    """Retourne les transformations de pré-traitement. À implémenter."""
    input_shape = config['model']['input_shape']
    target_size = (input_shape[1], input_shape[2]) # (224, 224)
    raise NotImplementedError("get_preprocess_transforms doit être implémentée par l'étudiant·e.")