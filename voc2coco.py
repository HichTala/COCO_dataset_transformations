import argparse
import json
import os
import re
from xml.etree import ElementTree as ET

from create_COCO_tree import create_coco_tree


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('root', default='', type=str)

    return parser.parse_args()


def get_label2id(labels_path):
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(1, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path, ann_ids_path, ext, ann_paths_list_path):
    # If use annotation paths list
    if ann_paths_list_path is not None:
        with open(ann_paths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotation ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def voc2coco(root):
    json_dict = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    bbox_id = 1
    categories = {}

    directories = ['train', 'val', 'test']
    for directory in directories:
        anno_path_list = os.path.join(root, f'ImageSets/Main/{directory}.txt')
        anno_dir = os.path.join(root, 'Annotations')

        anno_coco_dir, image_coco_dir = create_coco_tree(root, directory)

        with open(anno_path_list, 'r') as f:
            anno_list = f.read().split()

        for annotation in anno_list:
            anno_path = os.path.join(anno_dir, annotation + ".xml")
            annotation_tree = ET.parse(anno_path)
            annotation_root = annotation_tree.getroot()

            path = annotation_root.findtext('path')
            if path is None:
                file_name = annotation_root.findtext('filename')
            else:
                file_name = os.path.basename(path)

            image_path = os.path.join(root, 'JPEGImages', file_name)

            size = annotation_root.find('size')
            width = int(size.findtext('width'))
            height = int(size.findtext('height'))

            json_dict['images'].append({
                'file_name': file_name,
                'height': height,
                'width': width,
                'id': image_id
            })

            os.rename(image_path, image_coco_dir + file_name)

            for obj in annotation_root.findall('object'):
                category = obj.findtext('name')
                if category not in categories:
                    categories[category] = len(categories) + 1
                category_id = categories[category]

                bndbox = obj.find('bndbox')
                x_min = int(bndbox.findtext('xmin')) - 1
                y_min = int(bndbox.findtext('ymin')) - 1
                x_max = int(bndbox.findtext('xmax'))
                y_max = int(bndbox.findtext('ymax'))

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
    voc2coco(args.root)
