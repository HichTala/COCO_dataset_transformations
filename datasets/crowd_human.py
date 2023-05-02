# Original Code from https://blog.csdn.net/qq_41375609/article/details/95202218

import argparse
import json
import os

from PIL import Image

from datasets.create_COCO_tree import create_coco_tree


def parse_command_line():
    parser = argparse.ArgumentParser('Convert annotations to COCO format', add_help=False)

    parser.add_argument('root', default='', type=str)

    return parser.parse_args()


def crowdhuman2coco(root):
    directories = ['train', 'val']

    for directory in directories:
        image_dir = os.path.join(root, 'Images/')
        anno_path = os.path.join(root, f'annotation_{directory}.odgt')

        anno_coco_dir, image_coco_dir = create_coco_tree(root, directory)

        with open(anno_path, 'r') as file:
            lines = file.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]

        json_dict = {"images": [], "annotations": [], "categories": []}
        image_id = 1
        bbox_id = 1
        categories = {}
        record_list = len(records)
        print(record_list)

        for i in range(record_list):
            file_name = records[i]['ID'] + '.jpg'
            img = Image.open(image_dir + file_name)
            json_dict['images'].append({
                'file_name': file_name,
                'height': img.size[1],
                'width': img.size[0],
                'id': image_id
            })
            os.rename(image_dir + file_name, image_coco_dir + file_name)

            gt_box = records[i]['gtboxes']
            gt_box_len = len(gt_box)
            for j in range(gt_box_len):
                category = gt_box[j]['tag']
                if category not in categories:
                    categories[category] = len(categories) + 1
                category_id = categories[category]
                fbox = gt_box[j]['fbox']

                ignore = 0
                if "ignore" in gt_box[j]['head_attr']:
                    ignore = gt_box[j]['head_attr']['ignore']
                if "ignore" in gt_box[j]['extra']:
                    ignore = gt_box[j]['extra']['ignore']

                json_dict['annotations'].append({
                    "bbox": fbox,
                    "area": fbox[2] * fbox[3],
                    "segmentation": [],
                    "iscrowd": ignore,
                    "image_id": image_id,
                    "category_id": category_id,
                    "id": bbox_id
                })

                bbox_id += 1
            image_id += 1

        for category, cid in categories.items():
            json_dict['categories'].append({
                'supercategory': category,
                'id': cid,
                'name': category
            })

        json_file = open(anno_coco_dir + f'instances_{directory}2017.json', 'w')
        json_str = json.dumps(json_dict)
        json_file.write(json_str)
        json_file.close()


if __name__ == '__main__':
    args = parse_command_line()
    crowdhuman2coco(args.root)
