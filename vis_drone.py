import argparse
import json
import os

from PIL import Image


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('root', default='', type=str)

    return parser.parse_args()


def visdrone2coco(root):
    json_dict = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    bbox_id = 1
    categories = ['ignore', 'pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'awning-tricycle', 'tricycle',
                  'bus', 'motor', 'other']

    directories = ['train', 'val', 'test-dev']
    for directory in directories:
        image_dir = os.path.join(root, f'VisDrone2019-DET-{directory}/images/')
        anno_dir = os.path.join(root, f'VisDrone2019-DET-{directory}/annotations/')

        file_names = sorted(os.listdir(image_dir))
        for file_name in file_names:
            image = Image.open(image_dir + file_name)
            json_dict['images'].append({
                'file_name': file_name,
                'height': image.size[1],
                'width': image.size[0],
                'id': image_id
            })

            anno_path = os.path.join(anno_dir, os.path.splitext(file_name)[0] + '.txt')
            with open(anno_path) as f:
                annotations = f.read().split('\n')[:-1]
            annotations = [list(map(int, annotation.split(','))) for annotation in annotations]

            for annotation in annotations:
                if annotation[5] != 0 and annotation[5] != 1:
                    json_dict['annotations'].append({
                        "bbox": annotation[:4],
                        "area": annotation[2] * annotation[3],
                        "segmentation": [],
                        "iscrowd": 0,
                        "image_id": image_id,
                        "category_id": annotation[5],
                        "id": bbox_id
                    })
                    bbox_id += 1
            image_id += 1

            for id, category in enumerate(categories):
                json_dict['categories'].append({
                    "supercategory": category,
                    "id": id,
                    "name": category
                })

        if directory == 'test-dev':
            directory = directory[:4]
        anno_coco_path = os.path.join(root, f'annotations/instances_{directory}2017.json')
        image_coco_path = os.path.join(root, f'{directory}2017/')

        if not os.path.exists(anno_coco_path):
            os.mkdir(anno_coco_path)
        if not os.path.exists(image_coco_path):
            os.mkdir(image_coco_path)

        json_file = open(anno_coco_path, 'w')
        json_str = json.dumps(json_dict)
        json_file.write(json_str)
        json_file.close()

        os.rename(image_dir, image_coco_path)


if __name__ == '__main__':
    args = parse_command_line()
    visdrone2coco(args.root)
