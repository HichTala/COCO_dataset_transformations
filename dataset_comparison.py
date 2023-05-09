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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--save-path', default='./', type=str)

    return parser.parse_args()


def main(args):
    datasets = register(args.annotations)

    batch_size = args.batch_size

    for dataset in datasets:
        cfg = get_cfg()
        cfg.DATASETS.TRAIN = dataset
        cfg.SOLVER.IMS_PER_BATCH = batch_size

        dataloader = build_detection_train_loader(cfg)
        resnet = ResNet(cfg).cuda()

        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644" \
                            "/model_final_721ade.pkl"
        checkpointer = DetectionCheckpointer(resnet)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        batch_mean = []
        for data in dataloader:
            outputs = resnet(data)
            if len(data) == batch_size:
                outputs_mean = outputs.mean(0).unsqueeze(0)
                batch_mean.append(outputs_mean)
            else:
                outputs_mean = (outputs.sum(0) / batch_size).unsqueeze(0)
                batch_mean.append(outputs_mean)

        batch_mean = torch.cat(batch_mean).mean(0)
        save_path = os.path.join(args.save_path, dataset)

        with open(save_path, 'wb') as f:
            pickle.dump(batch_mean, f)
            print(dataset, "ok")


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
