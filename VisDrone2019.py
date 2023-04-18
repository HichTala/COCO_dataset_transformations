import argparse
import json
import os

from PIL import Image


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('root', default='', type=str)
    parser.add_argument('json_path', default='', type=str)

    return parser.parse_args()


def load_file(path):
    assert os.path.exists(path)

    with open(name + ".txt", 'r') as f:
        txt = f.read().split('\n')[:-1]
        txt = [list(map(int, t.split(','))) for t in txt]
    records = [json.loads(line.strip('\n')) for line in txt]
    return records


def visdrone2coco(root, json_path):
    json_dict = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    bbox_id = 1
    categories = ['ignore', 'pedestrian', 'person', 'bicycle', 'car', 'van', 'truck', 'awning-tricycle', 'tricycle',
                  'bus', 'motor', 'other']

    type = ['train', 'val', 'test-dev']
    for type in types:
        image_dir = os.path.join(root, f'VisDrone2019-DET-{type}/images')
        anno_dir = os.path.join(root, f'VisDrone2019-DET-{type}/annotations')

        file_names = sorted(os.listdir(image_dir))
        for file_name in file_names:
            image = Image.open(image_dir + file_name)
            json_dict['images'].append({
                'file_name': file_name,
                'height': image.size[1],
                'width': image.size[0],
                'id': image_id
            })

            anno_path = os.path.join(anno_dir, os.path.splitext(file_name)[0] + 'txt')
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

        json_file = open(json_path, 'w')
        json_str = json.dumps(json_dict)
        json_file.write(json_str)
        json_file.close()


if __name__ == '__main__':
    args = parse_command_line()
    visdrone2coco(args.root, args.json_path)
