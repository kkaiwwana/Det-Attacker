import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import torch
import torch.utils.data
from torch.utils.data import random_split
import torchvision
from PIL import Image
from pycocotools.coco import COCO


def imagenet_1k_mini(folder_path='./datasets/imagenet-mini', train_trans=None, val_trans=None):
    default_trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((256, 256)),
                                        transforms.CenterCrop([224, 224])])
    train_ds = ImageFolder(root=folder_path + '/train',
                           transform=train_trans if train_trans else default_trans)
    valid_ds = ImageFolder(root=folder_path + '/val',
                           transform=val_trans if val_trans else default_trans)
    return train_ds, valid_ds


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, min_size=(256, 256)):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.min_size = min_size

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        if self.transforms is not None:
            img = self.transforms(img)
        if img.shape[-1] < self.min_size[-1] or img.shape[-2] < self.min_size[-2]:
            return self._getitem(0)

        num_objs = len(coco_annotation)

        if num_objs == 0:
            return self._getitem(0)

        boxes = []
        labels = []
        for i in range(num_objs):
            x_min = coco_annotation[i]['bbox'][0]
            y_min = coco_annotation[i]['bbox'][1]
            x_max = x_min + coco_annotation[i]['bbox'][2]
            y_max = y_min + coco_annotation[i]['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(coco_annotation[i]['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        my_annotation = {"boxes": boxes, "labels": torch.tensor(labels), "image_id": torch.tensor([img_id])}

        return img, my_annotation

    def _getitem(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        if self.transforms is not None:
            img = self.transforms(img)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):

            x_min = coco_annotation[i]['bbox'][0]
            y_min = coco_annotation[i]['bbox'][1]
            x_max = x_min + coco_annotation[i]['bbox'][2]
            y_max = y_min + coco_annotation[i]['bbox'][3]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(coco_annotation[i]['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        my_annotation = {"boxes": boxes, "labels": torch.tensor(labels), "image_id": torch.tensor([img_id])}

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def coco_2017_dev_5k(
        folder_path='./datasets/COCO_dev/val2017/',
        annotation_path='./datasets/COCO_dev/annotations/instances_val2017.json',
        img_trans=None,
        requires_val=True,
        split_rate=None
):
    if requires_val and split_rate is None:
        split_rate = [0.8, 0.2]

    ds = COCODataset(folder_path, annotation_path, img_trans if img_trans else transforms.ToTensor())

    if requires_val:
        return random_split(ds, split_rate)
    else:
        return ds
