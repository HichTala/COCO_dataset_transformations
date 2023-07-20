# COCO_dataset_transformations


### Utilisation des scripts comparisons (intra-class)

Exemple:

```shell
python class_comparison.py datasets.json --save_path ./class_barycenter
```

#### Fonction des scripts

- `class_comparison` : Calcul moyenne + covariance pour chaques classes de chaques datasets
- `dataset_comparison` : Calcul moyenne + covariance pour chaque datasets
- `class_features` : Calcul moyenne + features de 10 images par classes pour chaque datasets
- `dataset_features` : Calcul moyenne + features de 10 images par datasets

### Utilisation des scripts comparisons (inter-class)

Exemple:

```shell
python inter-class_comparison.py datasets.json DIOR_train DOTA_train --save_path ./class_barycenter
```

$$ \frac{1}{|I_{c}||I_{c'}|} \sum_{i\inI_{c}, j\inI{c'} (\phi_{c}^{i}-\mu_{c})^\intercal (\phi_{c'}^{j}-\mu_{c'})$$

_Calcule un scalaire par couple de classes_