import argparse
import os

from pycocotools.coco import COCO


def parse_command_line():
    parser = argparse.ArgumentParser('Split into train/test/val', add_help=False)

    parser.add_argument('anntations', default='', type=str)
    parser.add_argument('source', default='', type=str)
    parser.add_argument('destination', default='', type=str)

    return parser.parse_args()


def main(annotations, source, destination):
    coco = COCO(annotations)

    for id in coco.getImgIds():
        im = coco.loadImgs(id)[0]
        source_path = os.path.join(source, im['file_name'])
        destination_path = os.path.join(destination, im['file_name'])

        os.rename(source_path, destination_path)


if __name__ == '__main__':
    args = parse_command_line()
    main(args.anntations, args.source, args.destination)
