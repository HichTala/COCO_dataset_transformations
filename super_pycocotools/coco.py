import json
import os

from pycocotools.coco import COCO as PYCOCO


class _PyCOCO:
    def __init__(self, info_dict=None):
        self.infos = info_dict

        ann_file = info_dict["path"] + '/annotations/instances_{}.json'
        if os.path.exists(ann_file.format('train2017')):
            print('train data')
            self.train = PYCOCO(ann_file.format('train2017'))
        if os.path.exists(ann_file.format('val2017')):
            print('val data')
            self.val = PYCOCO(ann_file.format('val2017'))
        if os.path.exists(ann_file.format('test2017')):
            print('test data')
            self.test = PYCOCO(ann_file.format('test2017'))


class COCO:
    def __init__(self, annotation_file=None):
        super_dataset = json.load(open(annotation_file, 'r'))
        assert type(super_dataset) == dict, "annotation file format {} not supported".format(type(super_dataset))

        self.datasets = super_dataset["datasets_name"]

        for dataset_name in super_dataset["datasets_name"]:
            print(f"processing {dataset_name}")
            pycoco = _PyCOCO(super_dataset["datasets_infos"][dataset_name])
            setattr(self, dataset_name, pycoco)
