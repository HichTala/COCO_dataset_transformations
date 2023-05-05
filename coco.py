import os

from PIL import Image
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, img_dir, coco, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.coco = coco
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)

        img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        # print(img.shape)

        return img, coco_annotation

    def __len__(self):
        return len(self.ids)
