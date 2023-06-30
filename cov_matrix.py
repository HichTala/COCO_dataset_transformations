import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def KL_divergence(mean1, mean2, covariance1, covariance2):
    assert covariance1.shape == covariance2.shape
    d = covariance1.shape[0]
    mean1 = mean1.type(torch.float64)
    mean2 = mean2.type(torch.float64)
    covariance1 = covariance1.type(torch.float64)
    covariance2 = covariance2.type(torch.float64)
    product = covariance2 @ torch.linalg.inv(covariance1)
    scal_prod = torch.t(mean2 - mean1) @ torch.linalg.inv(covariance2) @ (mean2 - mean1)
    return 1 / 2 * torch.logdet(product) - d / 2 + 1 / 2 * torch.trace(product) + 1 / 2 * scal_prod


def KL1(covariance1, covariance2):
    assert covariance1.shape == covariance2.shape
    d = covariance1.shape[0]
    covariance1 = covariance1.type(torch.float64)
    covariance2 = covariance2.type(torch.float64)
    product = covariance2 @ torch.linalg.inv(covariance1)
    return 1 / 2 * torch.logdet(product) - d / 2


def KL2(covariance1, covariance2):
    assert covariance1.shape == covariance2.shape
    covariance1 = covariance1.type(torch.float64)
    covariance2 = covariance2.type(torch.float64)
    product = covariance2 @ torch.linalg.inv(covariance1)
    return 1 / 2 * torch.trace(product)


def KL3(mean1, mean2, covariance2):
    assert covariance1.shape == covariance2.shape
    mean1 = mean1.type(torch.float64)
    mean2 = mean2.type(torch.float64)
    covariance2 = covariance2.type(torch.float64)
    scal_prod = torch.t(mean2 - mean1) @ torch.linalg.inv(covariance2) @ (mean2 - mean1)
    return 1 / 2 * scal_prod


datasets = os.listdir("./class_barycenter")

for dataset in datasets:
    path = os.path.join("./class_barycenter", dataset)
    mean = [x for x in sorted(os.listdir(path),
                              key=lambda item: (int(item.partition('_')[0])
                                                if item[0].isdigit() else float('inf'), item))
            if x.split('_')[1] == 'mean.pkl']

    cov = [x for x in sorted(os.listdir(path),
                             key=lambda item: (int(item.partition('_')[0])
                                               if item[0].isdigit() else float('inf'), item))
           if x.split('_')[1] == 'cov.pkl']

    n = len(mean)
    heatmap = np.zeros((n, n))
    heatmap1 = np.zeros((n, n))
    heatmap2 = np.zeros((n, n))
    heatmap3 = np.zeros((n, n))

    print(dataset)
    for i in tqdm(range(n)):
        mean1_path = os.path.join(path, mean[i])
        cov1_path = os.path.join(path, cov[i])
        with open(mean1_path, 'rb') as f:
            mean1 = pickle.load(f)
        with open(cov1_path, 'rb') as f:
            covariance1 = pickle.load(f)
        for j in range(n):
            mean2_path = os.path.join(path, mean[j])
            cov2_path = os.path.join(path, cov[j])
            with open(mean2_path, 'rb') as f:
                mean2 = pickle.load(f)
            with open(cov2_path, 'rb') as f:
                covariance2 = pickle.load(f)

            heatmap[i, j] = KL_divergence(mean1, mean2, covariance1, covariance2)
            heatmap1[i, j] = KL1(covariance1, covariance2)
            heatmap2[i, j] = KL2(covariance1, covariance2)
            heatmap3[i, j] = KL3(mean1, mean2, covariance2)

    save_folder = os.path.join("./heatmaps", dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, "heatmap.pkl"), "wb") as f:
        pickle.dump(heatmap, f)

    with open(os.path.join(save_folder, "heatmap1.pkl"), "wb") as f:
        pickle.dump(heatmap1, f)
    with open(os.path.join(save_folder, "heatmap2.pkl"), "wb") as f:
        pickle.dump(heatmap2, f)
    with open(os.path.join(save_folder, "heatmap3.pkl"), "wb") as f:
        pickle.dump(heatmap3, f)
