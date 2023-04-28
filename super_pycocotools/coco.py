import json
import os

from pycocotools.coco import COCO as PYCOCO


class _PyCOCO:
    def __init__(self, info_dict, name):
        self.infos = info_dict
        self.name = name

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

    def register(self):
        from detectron2.data.datasets import register_coco_instances

        data_dir = self.infos["path"]

        register_coco_instances(
            self.name + "_train",
            {},
            "instances_train2017.json",
            data_dir + "/train2017"
        )
        registered_dataset = (self.name + "_train",)
        if os.path.exists(self.infos["path"] + "/annotations/instances_val2017.json"):
            register_coco_instances(
                self.name + "_val",
                {},
                "json_annotation_val2017.json",
                data_dir + "/val2017"
            )
            registered_dataset += (self.name + "_val",)
        if os.path.exists(self.infos["path"] + "/annotations/instances_test2017.json"):
            register_coco_instances(
                self.name + "_test",
                {},
                "json_annotation_test2017.json",
                data_dir + "/test2017"
            )
            registered_dataset += (self.name + "_test",)
        return registered_dataset


class COCO:
    def __init__(self, annotation_file=None):
        self.super_dataset = json.load(open(annotation_file, 'r'))
        assert type(self.super_dataset) == dict, \
            "annotation file format {} not supported".format(type(self.super_dataset))

        self.datasets = self.super_dataset["datasets_name"]

        for dataset_name in self.datasets:
            print(f"processing {dataset_name}")
            pycoco = _PyCOCO(self.super_dataset["datasets_infos"][dataset_name], dataset_name)
            setattr(self, dataset_name, pycoco)

    def detectron_register(self, datasets=None):
        from detectron2.data.datasets import register_coco_instances

        if datasets is not None:
            dataset_names = datasets
        else:
            dataset_names = self.datasets

        registered_dataset_list = []

        for dataset_name in dataset_names:
            assert dataset_name in self.datasets, "unrecognized dataset {}".format(dataset_name)

            data_dir = self.super_dataset["datasets_infos"][dataset_name]["path"]

            register_coco_instances(
                dataset_name + "_train",
                {},
                "instances_train2017.json",
                data_dir + "/train2017"
            )
            registered_dataset = (dataset_name + "_train",)
            if os.path.exists(
                    self.super_dataset["datasets_infos"][dataset_name]["path"] + "/annotations/instances_val2017.json"
            ):
                register_coco_instances(
                    dataset_name + "_val",
                    {},
                    "instances_val2017.json",
                    data_dir + "/val2017"
                )
                registered_dataset += (dataset_name + "_val",)
            if os.path.exists(
                    self.super_dataset["datasets_infos"][dataset_name]["path"] + "/annotations/instances_test2017.json"
            ):
                register_coco_instances(
                    dataset_name + "_test",
                    {},
                    "instances_test2017.json",
                    data_dir + "/test2017"
                )
                registered_dataset += (dataset_name + "_test",)
            registered_dataset_list.append(registered_dataset)
        return registered_dataset_list
