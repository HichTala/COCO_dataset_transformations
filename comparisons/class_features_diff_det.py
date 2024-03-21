import argparse
import json
import os
import pickle

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import default_setup
from tqdm import tqdm

from diffusiondet import add_diffusiondet_config, add_fs_config, add_additional_config, create_unique_output_path
from diffusiondet.util.model_ema import add_model_ema_configs
from diffusiondet.train import FineTuningTrainer
from diffusiondet.data import register_dataset, LOCAL_CATALOG

from resnet_roi_pool import DiffDetResNetROIPool
from super_pycocotools.detectron import register


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('--study', type=str)
    parser.add_argument('--name', type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--dataset', type=str)

    return parser.parse_args()


def build_cfg_list_from_exp_file(study_file):
    with open(study_file, 'r') as f:
        study_json = json.load(f)

    study_names = study_json['names']
    study_dict = {}
    if len(study_names) == 1:
        study_names = study_names * len(study_json['studies'])

    for study_name, study in zip(study_names, study_json['studies']):
        seed_list = study_json['seed'] if 'seed' in study_json else [None]
        for seed in seed_list:
            n_values = [len(values) if isinstance(values, list) else 1 for param, values in study.items()]
            n_exp = max(n_values)
            assert all([v == n_exp or v == 1 for v in
                        n_values]), 'Inside one study, the number of different value for a parameter must be either 1 or n_exp'
            study_dict[study_name] = []
            for i in range(n_exp):
                exp_list = []
                for param, values in study.items():
                    exp_list.append(param)
                    if isinstance(values, list):
                        exp_list.append(values[i])
                    else:
                        exp_list.append(values)
                if seed is not None:
                    exp_list.append("SEED")
                    exp_list.append(seed)
                study_dict[study_name].append(exp_list)
    return study_dict


def setup(args, cfg_file, study_name):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_fs_config(cfg)
    add_additional_config(cfg)

    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = create_unique_output_path(cfg.OUTPUT_DIR, study_folder=study_name)

    return cfg


def main(args):
    # datasets = register(args.annotations)
    study_dict = build_cfg_list_from_exp_file(args.study)
    study_name = args.name
    cfg_list = study_dict[study_name]

    study_cfg = cfg_list[0]

    cfg_file = study_cfg[1]
    study_cfg = study_cfg[2:]
    cfg = setup(args, cfg_file, study_name)
    cfg.merge_from_list(study_cfg)
    # cfg.freeze()

    default_setup(cfg, args)

    register_dataset(LOCAL_CATALOG[cfg.DATASETS.TRAIN[0].split('_')[0]])

    cfg.SOLVER.IMS_PER_BATCH = 1

    dataloader = build_detection_train_loader(cfg)
    pool = DiffDetResNetROIPool(cfg).cuda()

    dataset_size = len(dataloader.dataset.dataset.dataset)

    # cfg.MODEL.WEIGHTS = "detectron2://backbone_cross_domain/model_final_721ade.pkl"
    cfg.MODEL.WEIGHTS = args.weights
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
            save_folder = os.path.join(args.save_path, args.name)
            save_class_path = os.path.join(args.save_path, args.name, str(class_id))

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if not os.path.exists(save_class_path):
                os.makedirs(save_class_path)

            # for i in tqdm(range(len(class_mean[class_id][:10])), desc=f"Processing class {class_id}", colour='green'):
            save_path = os.path.join(args.save_path, args.name, str(class_id), 'features.pth')
            torch.save(class_mean[class_id], save_path)
            # with open(save_path, 'wb') as f:
            #     pickle.dump(class_mean[class_id], f)

            save_path_mean = os.path.join(args.save_path, args.name, str(class_id), 'mean.pth')

            mean = torch.cat(class_mean[class_id]).mean(0)
            torch.save(mean, save_path_mean)


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
