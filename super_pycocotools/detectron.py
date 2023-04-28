import json
import os

from detectron2.data.datasets import register_coco_instances


def register(dataset, annotation_file):
    super_dataset = json.load(open(annotation_file, 'r'))

    assert dataset in super_dataset["datasets_name"], "unrecognized dataset {}".format(dataset)

    data_dir = super_dataset["datasets_infos"][dataset]["path"]

    register_coco_instances(
        dataset + "_train",
        {},
        "instances_train2017.json",
        data_dir + "/train2017"
    )
    registered_datasets = (dataset + "_train",)
    if os.path.exists(super_dataset["datasets_infos"][dataset]["path"] + "/annotations/instances_val2017.json"):
        register_coco_instances(
            dataset + "_val",
            {},
            "json_annotation_val2017.json",
            data_dir + "/val2017"
        )
        registered_datasets += (dataset + "_val",)
    if os.path.exists(super_dataset["datasets_infos"][dataset]["path"] + "/annotations/instances_test2017.json"):
        register_coco_instances(
            dataset + "_test",
            {},
            "json_annotation_test2017.json",
            data_dir + "/test2017"
        )
        registered_datasets += (dataset + "_test",)

    return registered_datasets

