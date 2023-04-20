import argparse
import json
import os

import numpy as np
from PIL import Image

from create_COCO_tree import create_coco_tree


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('root', default='', type=str)

    return parser.parse_args()


def deep_fruits(root):
    data_type = ['train', 'test']
    for data in data_type:
        json_dict = {"images": [], "annotations": [], "categories": []}
        categories = {}
        image_id = 1
        bbox_id = 1

        for category in os.listdir(root):
            directory = os.path.join(root, category)

            anno_path_list = os.path.join(directory, f'{data}_RGB.txt')
            image_dir = os.path.join(directory, f'{data.upper()}_RGB/')

            with open(anno_path_list, 'r') as f:
                annotations = f.read().split('\n')

            for annotation in annotations:
                annotation = annotation.split()

                file_name = annotation[0].split('/')[-1]
                image = Image.open(image_dir + file_name)
                json_dict['images'].append({
                    'file_name': file_name,
                    'height': image.size[1],
                    'width': image.size[0],
                    'id': image_id
                })

                bboxs = np.split(np.array(annotation[2:]), annotation[1])
                for bbox in bboxs:
                    x_min, y_min, x_max, y_max = list(map(int, bbox[:4]))
                    category_id = int(bbox[4])

                    json_dict['annotations'].append({
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "segmentation": [],
                        "iscrowd": 0,
                        "image_id": image_id,
                        "category_id": category_id,
                        "id": bbox_id
                    })

                    bbox_id += 1
                image_id += 1

                json_dict['categories'].append({
                    'supercategory': category,
                    'id': category_id,
                    'name': category
                })


if __name__ == '__main__':
    args = parse_command_line()
    deep_fruits(args.root)
