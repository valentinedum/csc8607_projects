# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : _DUMANGE, Valentine_
- **Projet** : _CUB-200-2011 (oiseaux, 200 espèces) avec convolutions groupées (grouped convolutions) (CUB-200-2011 × CNN 2D à 3 stages)_
- **Dépôt Git** : _[URL publique](https://github.com/valentinedum/csc8607_projects)_
- **Environnement** : `python == 3.10.18`, `torch == 2.5.1`, `cuda == 12.1`  
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset
- **Source** (lien) : https://huggingface.co/datasets/dpdl-benchmark/caltech_birds2011
- **Type d’entrée** (image / texte / audio / séries) : images et textes
- **Tâche** (multiclasses, multi-label, régression) : classification multiclasses
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : images de tailles variées, après normalisation -> 224×224
- **Nombre de classes** (`meta["num_classes"]`) :  200 classes d’oiseaux

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

Nous allons utiliser le dataset de huggingface nommé caltech_birds2011 (url ci-dessus). C'est un dataset avec des images d'oiseaux ainsi que leur espece (texte). Les images sont de tailles variables.

### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
| ----: | --------: | ----------------------------------------------------- |
| Train |      4795 |         équilibré, pas de labels manquants,     images de tailles variées                                      |
|   Val |      1199 |           équilibré, pas de labels manquants, iamges de tailles variées                                           |
|  Test |      5794 |               déséquilibré,    pas de labels manquants , images de tailles variées                                   |

**D2.** Donnez la taille de chaque split et le nombre de classes.  
Il y a 200 classes dans chaque split et le train fait 4795 lignes, le validation 1199, le test 5794.

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).
Dans mon dataset, j'avais initialement aucun set de validation. J'ai donc créé mon propre split à partir du train en appliquant une stratification par classe avec un ratio de 80% train / 20% val.
La seed a été fixée à 42 pour permettre la reproductibilité.

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.  
La distribution des classes montrent que les ensembles Train et Val sont très équilibrés, il ya autant d'échantillons (24) dans chaque classe. A l'inverse le dataset Test est moins équilibré. La plupart des classes ont 30 échantillons mais certains en ont moins avec 12 éachntillons, ou une vingtaine.
Des graphiques ont été ploté via tensorboard (voir le dossier runs).
![distribution_train](./artifacts/distribution_train.png)
![distribution_test](./artifacts/distribution_test.png)
![distribution_val](./artifacts/distribution_val.png)

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).
Ce dataset a très peu de particularités. Il n'a aucun label manquant. Ces images sont toutes en RGB mais sont par contres de tailles très différentes, on compte au moins 750 différentes tailles d'images.

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- Vision : resize = [224, 224], center-crop = None, normalize = (mean=[0.48185426, 0.50031734, 0.42832923], std=[0.2270571, 0.2226704, 0.26213554])…
- Audio : resample = __ Hz, mel-spectrogram (n_mels=__, n_fft=__, hop_length=__), AmplitudeToDB…
- NLP : tokenizer = __, vocab = __, max_length = __, padding/truncation = __…
- Séries : normalisation par canal, fenêtrage = __…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ? 
Comme indiqué en D5 , les images du dataset ont des tailles très variés. Seulement es réseaux de neurones convolutifs ont besoin d'une entrée à taille fixe. Nous redimensionnons donc les images à [224, 224]. Après le resizing, il est important de transformer l'image en tenseur pour pouvoir la traiter avec pytorch. Puis nous normalisons notre tenseur avec mean=[0.48185426, 0.50031734, 0.42832923] et std=[0.2270571, 0.2226704, 0.26213554] car après analyse ce sont les statistiques que nous avons à propos du dataset d'entrainement. Autrement nous aurions pu trouver sur internet les paramètres moyens des datasets d'images connues tels que ImageNet et approximer par ceux-ci.
NB: Je n'ai pas fait de center-crop car un redimensionnement de l'image avait déjà été fait. On ne voudrait pas qu'une partie de l'oiseau soit accidentellement coupée.

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ? Tous mes prétraitements sont les mêmes pour train, val et test pour être sur que l'évaluation sera représentative. Toutefois train recevra en plus de stransformations d'augmentation de données.

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - ex. Flip horizontal p=0.5, RandomResizedCrop scale=__, ratio=__ …
  - Audio : time/freq masking (taille, nb masques) …
  - Séries : jitter amplitude=__, scaling=__ …

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?  
Nous avons appliqué à notre dataset plusieurs augmentations car nous avons un grand risque de surapprentissage avec très peu d'images par classe (24 environ). Pour que le modèle devienne plus robuste, nous appliquons un randomHorizontalFlip de probabilité 0.5 car un oiseau est le même qu'il soit tourné vers la droite ou la gauche. Nous allons ainsi rendre le modèle invariant à cela.
Nous allons aussi prendre en compte le fait que les photos ont pu être prises sous différentes conditions d'éclairage. Les paramètres Variations aléatoires de luminosité (±20%), contraste (±20%), saturation (±20%) et teinte (±10%) ont été choisi d'apès la documentation de pytorch.
Enfin nous allons aussi appliquer des petites rotations (+-15 degrés) car les oiseaux peuvent être plus ou moins inclinés. On rendra ainsi le modèle robuste aux changements d'orientation.

**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.
Oui, les transformations conservent les labels, c'est bien le plus important. Le modèle doit comprendre qui oiseau qu'il ait la tête à droite ou à gauche, qu'il ait été pris plus ou moins au soleil et plus ou moins incliné est le même. Il garde la même espèce.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> ![original_0](./artifacts/original_0.png)
> ![augmented_0](./artifacts/augmented_0.png)
> ![preprocessed_0](./artifacts/preprocessed_augmented_0.png)

**D10.** Montrez 2–3 exemples et commentez brièvement.  
Les 3 images au-dessus sont de haut en bas:

- l'image originale
- l'image après augmentation
- l'image après augmentation puis preprocessing

Avec cet exemple, nous remarquons en effet que l'image avait une chance sur deux de connaitre une symétrie horizontale lors de l'augmentation. Ca n'a pas été le cas. Sinon, elle a un petit peu changé en termes de couleurs et a connu une petite rotation (<15 degrès en effet). L'augmentation semble avoir bien fonctionné.

Puis le preprocessing a normalisé les couleurs de l'image d'où le changement marquant et surtout on a redimensionné l'image afin qu'elle soit en format carré de 224 par 224. Le preprocessing a lui aussi bien fonctionné.

**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.
D'après la sortie de mon script de test dans data_loading, la forme exact de sortie d'un batch est (64, 3, 224, 224). Ce qui est cohérent avec le "batch_size" et "input_shape" inscrit dans les configs.

---

## 2) Modèle

### 2.1 Baselines

**M0.**
- **Classe majoritaire** — Métrique : `Accuracy` → score = `0.52%`
- **Prédiction aléatoire uniforme** — Métrique : `Accuracy` → score = `0.41%`  
_Ces scores très faibles (proches de 1/200 = 0.5%) confirment que le dataset est équilibré et ne présentent pas de biais. Notre modèle devra faire mieux que ces scores en apprenant des caractéristiques discriminantes_

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :
  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :
  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = __(batch_size, num_classes)__ (ou __(batch_size, num_attributes)__)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).


### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :
  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR  | WD  | Hyp-A | Hyp-B | Val metric (nom=_____) | Val loss | Notes |
| ------------------- | --- | --- | ----- | ----- | ---------------------- | -------- | ----- |
|                     |     |     |       |       |                        |          |       |
|                     |     |     |       |       |                        |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_

- **Variation du LR** (impact au début d’entraînement)
- **Variation du weight decay** (écart train/val, régularisation)
- **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
````

* **Artifacts requis présents** :

  * [ ] `runs/` (runs utiles uniquement)
  * [ ] `artifacts/best.ckpt`
  * [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

* PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
* Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
* Toute ressource externe substantielle (une ligne par référence).


