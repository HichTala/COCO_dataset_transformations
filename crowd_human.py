import json
import os

from PIL import Image

def parse_command_line():
    parser = argparse.ArgumentParser('parser', add_help=False)

    parser.add_argument('odgt_path', default='', type=str)
    parser.add_argument('json_path', default='', type=str)

    return parser.parse_args()


def load_file(path):
    assert os.path.exists(path)

    with open(path, 'r') as file:
        lines = file.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]
    return records


def crowdhuman2coco(odgt_path, json_path):
    records = load_file(odgt_path)

    json_dict = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    bbox_id = 1
    categories = {}
    record_list = len(records)
    print(record_list)

    for i in range(record_list):
        file_name = records[i]['ID'] + '.jpg'
        img = Image.open("/gpfsscratch/rech/vlf/ues92cf/CrowdHuman/Images" + file_name)
        image = {'file_name': file_name, 'height': img.size[1], 'width': img.size[0], 'id': image_id}
        json_dict['images'].append(image)

        gt_box = records[i]['gt_boxes']
        gt_box_len = len(gt_box)
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id
            category_id = categories[category]
            fbox = gt_box[j]['fbox']

            ignore = 0
            if "ignore" in gt_box[j]['head_attr']:
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']
            annotations = {'area': fbox[2] * fbox[3], 'is_crowd': ignore, 'image_id': image_id, 'bbox': fbox,
                           'hbox': gt_box[j]['hbox'], 'vbox': gt_box[j]['vbox'], 'category_id': category_id,
                           'id': bbox_id, 'ignore': ignore, 'segmentation': []}
            json_dict['annotations'].append(annotations)

            bbox_id += 1
        image_id += 1


    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

    print("Work done!")

if __name__ == '__main__':
    args = parse_command_line()
    crowdhuman2coco(args.odgt_path, args.json_path)