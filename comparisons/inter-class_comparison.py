import argparse
import os
import pickle

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader

from comparisons.resnet_roi_pool import ResNetROIPool
from super_pycocotools.detectron import register


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('annotations', default='', type=str)
    parser.add_argument('dataset1', default='', type=str)
    parser.add_argument('dataset2', default='', type=str)
    parser.add_argument('--save-path', default='./', type=str)

    return parser.parse_args()


def main(args):
    datasets = register(args.annotations)

    batch_size = 1

    dataset1 = args.dataset1
    dataset2 = args.dataset2

    cfg1 = get_cfg()
    cfg1.DATASETS.TRAIN = dataset1
    cfg1.SOLVER.IMS_PER_BATCH = batch_size

    cfg2 = get_cfg()
    cfg2.DATASETS.TRAIN = dataset2
    cfg2.SOLVER.IMS_PER_BATCH = batch_size

    dataloader1 = build_detection_train_loader(cfg1)
    dataloader2 = build_detection_train_loader(cfg2)
    pool = ResNetROIPool(cfg1).cuda()

    dataset_size1 = len(dataloader1.dataset.dataset.dataset)
    dataset_size2 = len(dataloader2.dataset.dataset.dataset)

    cfg1.MODEL.WEIGHTS = "detectron2://backbone_cross_domain/model_final_721ade.pkl"
    cfg2.MODEL.WEIGHTS = "detectron2://backbone_cross_domain/model_final_721ade.pkl"
    checkpointer1 = DetectionCheckpointer(pool)
    checkpointer2 = DetectionCheckpointer(pool)
    checkpointer1.load(cfg1.MODEL.WEIGHTS)
    checkpointer2.load(cfg2.MODEL.WEIGHTS)

    class_mean1 = {}
    class_mean2 = {}
    with torch.no_grad():
        dataloader_iteration1 = iter(dataloader1)
        for i in range(dataset_size1):
            data = next(dataloader_iteration1)
            box_features, target_classes = pool(data)

            for target_classe in target_classes:
                for class_id, box in zip(target_classe, box_features):
                    if class_id.item() not in class_mean1:
                        class_mean1[class_id.item()] = []
                    class_mean1[class_id.item()].append(box.unsqueeze(dim=0))

        dataloader_iteration2 = iter(dataloader2)
        for i in range(dataset_size2):
            data = next(dataloader_iteration2)
            box_features, target_classes = pool(data)

            for target_classe in target_classes:
                for class_id, box in zip(target_classe, box_features):
                    if class_id.item() not in class_mean2:
                        class_mean2[class_id.item()] = []
                    class_mean2[class_id.item()].append(box.unsqueeze(dim=0))

        for class_id1 in class_mean1.keys():
            mean1 = torch.cat(class_mean1[class_id1]).mean(0)

            for class_id2 in class_mean2.keys():
                mean2 = torch.cat(class_mean2[class_id2]).mean(0)

                sum_matrix1 = torch.cat(class_mean1[class_id1]).sum(0)
                sum_matrix2 = torch.cat(class_mean2[class_id2]).sum(0)

                cov_matrix = ((sum_matrix1 - dataset_size1 * mean1).t() @ (sum_matrix2 - dataset_size2 * mean2)) / (dataset_size1 * dataset_size2)

                save_folder = os.path.join(args.save_path, dataset1 + "x" + dataset2)
                save_path = os.path.join(args.save_path, dataset1 + "x" + dataset2,
                                         dataset1 + str(class_id1) + "x" + dataset2 + str(class_id2))

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
            #
            # with open(save_path + '_mean.pkl', 'wb') as f:
            #     pickle.dump(mean, f)
                with open(save_path + '_cov.pkl', 'wb') as f:
                    pickle.dump(cov_matrix, f)

            # print(dataset, "ok")


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
