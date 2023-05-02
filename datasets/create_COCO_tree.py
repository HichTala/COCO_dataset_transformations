import os


def create_coco_tree(root, directory):
    anno_coco_dir = os.path.join(root, 'annotations/')
    image_coco_dir = os.path.join(root, f'{directory}2017/')

    if not os.path.exists(anno_coco_dir):
        os.mkdir(anno_coco_dir)
    if not os.path.exists(image_coco_dir):
        os.mkdir(image_coco_dir)

    return anno_coco_dir, image_coco_dir
