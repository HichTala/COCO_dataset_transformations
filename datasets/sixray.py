import argparse
import json
import os
from xml.etree import ElementTree as ET

from create_COCO_tree import create_coco_tree


def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('root', default='', type=str)

    return parser.parse_args()


def voc2coco(root):
    image_id = 1
    bbox_id = 1

    directories = ['train', 'test']
    for directory in directories:
        json_dict = {"images": [], "annotations": [], "categories": []}
        categories = {}
        anno_path_list = os.path.join(root, f'ImageSet/100/{directory}.csv')
        anno_dir = os.path.join(root, 'Annotation')

        anno_coco_dir, image_coco_dir = create_coco_tree(root, directory)

        with open(anno_path_list, 'r') as f:
            anno_list = f.read().split()
        anno_list = [ann.split(',')[0] for ann in anno_list][1:]

        for annotation in anno_list:
            anno_path = os.path.join(anno_dir, annotation + ".xml")
            if os.path.exists(anno_path):
                annotation_tree = ET.parse(anno_path)
                annotation_root = annotation_tree.getroot()

                path = annotation_root.findtext('path')
                if path is None:
                    file_name = annotation_root.findtext('filename')
                else:
                    file_name = os.path.basename(path)

                if file_name[-4:] != '.jpg':
                    file_name = file_name + '.jpg'

                try:
                    image_path = os.path.join(root, 'JPEGImage', file_name)

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
                        if category is not None:
                            if category not in categories:
                                categories[category] = len(categories) + 1
                            category_id = categories[category]

                            bndbox = obj.find('bndbox')
                            x_min = int(float(bndbox.findtext('xmin'))) - 1
                            y_min = int(float(bndbox.findtext('ymin'))) - 1
                            x_max = int(float(bndbox.findtext('xmax')))
                            y_max = int(float(bndbox.findtext('ymax')))

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
                except FileNotFoundError:
                    print(file_name, "doesn't exists so the image will not be put in the dataset")

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
