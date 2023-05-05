import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader

from resnet import ResNet
from super_pycocotools.coco import SuperCOCO


def main():
    super_ann_file = './datasets.json'
    coco = SuperCOCO(super_ann_file)

    # data_dir = coco.SIXray.infos['root']
    # data_type = 'train2017'
    # img_dir = '{}/{}/'.format(data_dir, data_type)

    cfg = get_cfg()
    cfg.DATASETS.TRAIN = coco.SIXray.register()[0]
    cfg.SOLVER.IMS_PER_BATCH = 2

    dataloader = build_detection_train_loader(cfg)
    resnet = ResNet(cfg).cuda()

    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644" \
                        "/model_final_721ade.pkl"
    checkpointer = DetectionCheckpointer(resnet)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    batch_mean = []
    for data in dataloader:
        outputs = resnet(data).mean(0).unsqueeze(0)
        batch_mean.append(outputs)

    batch_mean = torch.cat(batch_mean)

    return batch_mean.mean(0)


if __name__ == '__main__':
    mean = main()
    print(mean)
