import argparse
import os
import pickle

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from tqdm import tqdm

from resnet_roi_pool import ResNetROIPool
from super_pycocotools.detectron import register


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('annotations', default='', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save-path', default='./', type=str)

    return parser.parse_args()


def main(args):
    datasets = register(args.annotations)

    batch_size = args.batch_size

    for dataset_tuple in datasets:
        for dataset in dataset_tuple:
            cfg = get_cfg()
            cfg.DATASETS.TRAIN = dataset
            cfg.SOLVER.IMS_PER_BATCH = batch_size

            dataloader = build_detection_train_loader(cfg)
            pool = ResNetROIPool(cfg).cuda()

            dataset_size = len(dataloader.dataset.dataset.dataset)

            # cfg.MODEL.WEIGHTS = "detectron2://backbone_cross_domain/model_final_721ade.pkl"
            cfg.MODEL.WEIGHTS = "./model_final_721ade.pkl"
            checkpointer = DetectionCheckpointer(pool)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            class_mean = {}
            with torch.no_grad():
                dataloader_iteration = iter(dataloader)
                for _ in tqdm(range(dataset_size), desc="Calculating mean", colour='cyan'):
                    data = next(dataloader_iteration)
                    box_features, target_classes = pool(data)

                    for target_classe in target_classes:
                        for class_id, box in zip(target_classe, box_features):
                            if class_id.item() not in class_mean:
                                class_mean[class_id.item()] = []
                            class_mean[class_id.item()].append(box.unsqueeze(dim=0))

                for class_id in class_mean.keys():
                    print(f"Processing class {class_id}", end='\r')
                    save_folder = os.path.join(args.save_path, dataset)
                    save_class_path = os.path.join(args.save_path, dataset, str(class_id))

                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    if not os.path.exists(save_class_path):
                        os.makedirs(save_class_path)

                    # for i in tqdm(range(len(class_mean[class_id][:10])), desc=f"Processing class {class_id}", colour='green'):
                    save_path = os.path.join(args.save_path, dataset, str(class_id), 'features.pth')
                    torch.save(class_mean[class_id], save_path)
                    # with open(save_path, 'wb') as f:
                    #     pickle.dump(class_mean[class_id], f)

                    save_path_mean = os.path.join(args.save_path, dataset, str(class_id), 'mean.pth')

                    mean = torch.cat(class_mean[class_id]).mean(0)
                    torch.save(mean, save_path_mean)

            print(dataset, "ok")


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
