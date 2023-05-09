import json
import warnings


def register(annotation_file, datasets=None):
    from detectron2.data.datasets import register_coco_instances

    super_dataset = json.load(open(annotation_file, 'r'))
    dataset_names = super_dataset["datasets_name"]

    if datasets is not None:
        if isinstance(datasets, list):
            dataset_names = datasets
        elif isinstance(datasets, str):
            dataset_names = [datasets]
        else:
            dataset_names = list(datasets)

    registered_dataset_list = []

    for dataset_name in dataset_names:
        try:
            root = super_dataset["datasets_infos"][dataset_name]["root"]
            train_annotation = super_dataset["datasets_infos"][dataset_name]["train"]

            register_coco_instances(
                dataset_name + "_train",
                {},
                train_annotation,
                root + "/train2017"
            )
            registered_dataset = (dataset_name + "_train",)
            if super_dataset["datasets_infos"][dataset_name]["val"] != "":
                val_annotation = super_dataset["datasets_infos"][dataset_name]["val"]
                register_coco_instances(
                    dataset_name + "_val",
                    {},
                    val_annotation,
                    root + "/val2017"
                )
                registered_dataset += (dataset_name + "_val",)
            if super_dataset["datasets_infos"][dataset_name]["test"] != "":
                test_annotation = super_dataset["datasets_infos"][dataset_name]["test"]
                register_coco_instances(
                    dataset_name + "_test",
                    {},
                    test_annotation,
                    root + "/test2017"
                )
                registered_dataset += (dataset_name + "_test",)
            registered_dataset_list.append(registered_dataset)
        except KeyError:
            warnings.warn("Dataset {} hasn't been registered: unrecognized dataset".format(dataset_name))
        except AssertionError as e:
            warnings.warn(str(e))
    return registered_dataset_list
