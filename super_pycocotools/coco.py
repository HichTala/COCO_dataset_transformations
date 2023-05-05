import json
import os

from pycocotools.coco import COCO as CocoSplit


class _CocoDataset:
    def __init__(self, info_dict, name):
        self.infos = info_dict
        self.name = name

        if self.infos["train"] != "":
            print('train data')
            self.train = CocoSplit(self.infos["train"])
        if self.infos["val"] != "":
            print('val data')
            self.val = CocoSplit(self.infos["val"])
        if self.infos["test"] != "":
            print('test data')
            self.test = CocoSplit(self.infos["test"])

    def register(self):
        from detectron2.data.datasets import register_coco_instances

        root = self.infos["root"]
        train_annotation = self.infos["train"]

        register_coco_instances(
            self.name + "_train",
            {},
            train_annotation,
            root + "/train2017"
        )
        registered_dataset = (self.name + "_train",)
        if self.infos["val"] != "":
            val_annotation = self.infos["val"]
            register_coco_instances(
                self.name + "_val",
                {},
                val_annotation,
                root + "/val2017"
            )
            registered_dataset += (self.name + "_val",)
        if self.infos["test"] != "":
            test_annotation = self.infos["test"]
            register_coco_instances(
                self.name + "_test",
                {},
                test_annotation,
                root + "/test2017"
            )
            registered_dataset += (self.name + "_test",)
        return registered_dataset


class SuperCOCO:
    def __init__(self, annotation_file=None):
        self.super_dataset = json.load(open(annotation_file, 'r'))
        assert type(self.super_dataset) == dict, \
            "annotation file format {} not supported".format(type(self.super_dataset))

        self.datasets = self.super_dataset["datasets_name"]

        for dataset_name in self.datasets:
            print(f"processing {dataset_name}")
            pycoco = _CocoDataset(self.super_dataset["datasets_infos"][dataset_name], dataset_name)
            setattr(self, dataset_name, pycoco)

    def __getitem__(self, key):
        return getattr(self, key)
