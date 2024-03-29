import argparse
import os
import pickle

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader

from resnet import ResNet
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
            resnet = ResNet(cfg).cuda()

            dataset_size = len(dataloader.dataset.dataset.dataset)

            cfg.MODEL.WEIGHTS = "detectron2://backbone_cross_domain/model_final_721ade.pkl"
            checkpointer = DetectionCheckpointer(resnet)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            batch_list = []
            with torch.no_grad():
                dataloader_iteration = iter(dataloader)
                for i in range(dataset_size):
                    data = next(dataloader_iteration)
                    outputs = resnet(data)
                    batch_list.append(outputs)

                batch_mean = torch.cat(batch_list).mean(0)
                cov_matrix = torch.cov(torch.cat(batch_list).t())
                save_path = os.path.join(args.save_path, dataset)

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)

                with open(save_path + '_mean.pkl', 'wb') as f:
                    pickle.dump(batch_mean, f)
                with open(save_path + '_std.pkl', 'wb') as f:
                    pickle.dump(cov_matrix, f)

            print(dataset, "ok")


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
