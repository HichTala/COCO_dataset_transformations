import os

import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt


def normalize_dict_values(d):
    min_val = min(d.values())
    max_val = max(d.values())

    normalized_dict = {}
    for key, val in d.items():
        normalized_val = (val - min_val) / (max_val - min_val)
        normalized_dict[key] = normalized_val

    return normalized_dict


datasets = ['DOTA_train', 'DOTA_test', 'DOTA_val', 'COCO_train', 'COCO_val', 'DIOR_train', 'DIOR_test',
            'DIOR_val', 'pascalvoc2012_train', 'pascalvoc2012_test', 'pascalvoc2012_val']
for dataset in datasets:
    classes = os.listdir(f'../{dataset}')

    mean_dict = {}
    features_dict = {}
    for c in tqdm.tqdm(classes):
        class_features = []
        mean = torch.load(os.path.join(f'../{dataset}', c, 'mean.pth'))
        mean_dict[c] = mean.cpu().numpy()

        image_features = torch.load(os.path.join(f'../{dataset}', c, 'features.pth'))

        for i, img_f in enumerate(image_features):
            class_features.append(img_f.squeeze().cpu().numpy())
        features_dict[c] = np.array(class_features)

    concentration_dict = {}
    for c in tqdm.tqdm(classes):
        concentration_dict[c] = 1 / features_dict[c].shape[0] * (((features_dict[c] - mean_dict[c]) ** 2).sum()) ** 0.5

    concentration_dict = normalize_dict_values(concentration_dict)

    plt.bar(concentration_dict.keys(), concentration_dict.values())
    plt.savefig("../figures/barplot_{dataset}.png".format(dataset=dataset))

    distances = np.zeros((len(classes), len(classes)))
    for ci in classes:
        for cj in classes:
            distances[int(ci), int(cj)] = (((mean_dict[ci] - mean_dict[cj]) ** 2).sum()) ** 0.5

    distances = distances / distances.max()

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    im = ax.imshow(distances)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    plt.savefig("../figures/heatmap_{dataset}.png".format(dataset=dataset))

    clustering_matrix = np.zeros((len(classes), len(classes)))
    for ci in classes:
        for cj in classes:
            if ci == cj:
                clustering_matrix[int(ci), int(cj)] = 0
            else:
                clustering_matrix[int(ci), int(cj)] = (concentration_dict[ci] + concentration_dict[cj]) / distances[
                    int(ci), int(cj)]

    clustering = {}
    for c in classes:
        clustering[c] = max(clustering_matrix[int(c)])
    plt.bar(clustering.keys(), clustering.values())
    plt.savefig("../figures/clustering_{dataset}.png".format(dataset=dataset))

    db = np.array(list(clustering.values())).sum() / len(classes)
    print(dataset, db)
