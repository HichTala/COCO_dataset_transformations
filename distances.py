import pickle
import torch.nn.functional as F


def main():
    with open('filename.pkl', 'rb') as f:
        coco = pickle.load(f)


mean = 5
