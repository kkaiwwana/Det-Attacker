import random

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from typing import *
from tqdm import tqdm


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
    def __init__(self, root, annotation, img_trans=None, target_trans=None, min_size=(256, 256)):
        self.root = root
        self.img_trans = img_trans
        self.target_trans = target_trans
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
        origin_size = torch.tensor(img.size)
        if self.img_trans is not None:
            img = self.img_trans(img)
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

        my_annotation = {'boxes': boxes,
                         'labels': torch.tensor(labels),
                         'image_id': torch.tensor([img_id]),
                         'origin_size': origin_size}

        if self.target_trans:
            my_annotation = self.target_trans(my_annotation)

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
        origin_size = torch.tensor(img.size)

        if self.img_trans is not None:
            img = self.img_trans(img)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [x_min, y_min, width, height]
        # In pytorch, the input should be [x_min, y_min, x_max, y_max]
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

        my_annotation = {'boxes': boxes,
                         'labels': torch.tensor(labels),
                         'image_id': torch.tensor([img_id]),
                         'origin_size': origin_size}
        if self.target_trans:
            my_annotation = self.target_trans(my_annotation)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# TODO 1: overwrite random_split function, customize it to be able to set the threshold of num objects in the image,
#  which will be add to train dataset. -> done

# TODO 2: write a activate function like ReLU but different threshold for Patch Generator (cuz normal ReLU does suit.
#  ) -> done

# TODO 3: write a fine-tune method for patch generator(
#  to train a patch that has been trained, with stronger augmentation maybe.) -> done

# TODO 4: maybe we need a weight-decay. but the normal version doest suit here, so make one by myself. -> done

# TODO 5: implement Fast Gradient Sign Method, for potential need. -> done

# TODO 6: YOLO, pytorch
# TODO 7: VOC, pytorch


def coco_2017_dev_5k(
        folder_path='./datasets/COCO_dev/val2017/',
        annotation_path='./datasets/COCO_dev/annotations/instances_val2017.json',
        img_trans=None,
        target_trans=None,
        requires_val=True,
        split_rate=None,
        data_selectors=None
):
    def selective_split_coco(
            dataset: Sized and data.Dataset,
            lengths: List[Union[int, float]],
            selectors: List[Callable] = None
    ) -> List[data.Subset]:

        len_dataset = len(dataset)
        indices = list(range(len_dataset))
        random.shuffle(indices)

        if isinstance(lengths[0], float):
            assert sum(lengths) - 1.0 > 1e-4, 'sum of subset lengths should be 1.0 while passing fractions.'
            for i in range(len(lengths)):
                lengths[i] *= len_dataset
        elif isinstance(lengths[0], int):
            assert sum(lengths) <= len_dataset, 'sum of subset lengths shouldn\' grater than origin dataset.'

        assert len(selectors) == len(lengths), 'num of selectors should be equal to num of subsets.'

        subsets_indices = {i: {'indices': [], 'length': length}for i, length in enumerate(lengths)}

        with tqdm(total=len_dataset) as pbar:
            pbar.set_description('Selecting data')

            for idx in indices:
                for i, selector in enumerate(selectors):
                    if selector is None or selector(dataset[idx]):
                        if len(subsets_indices[i]['indices']) < subsets_indices[i]['length']:
                            subsets_indices[i]['indices'].append(idx)
                            break
                pbar.update(1)

        for subset in subsets_indices.values():
            if len(subset['indices']) < subset['length']:
                assert False, 'no enough data satisfy the requirement.'

        return [(data.Subset(dataset, subset['indices']) if len(subset['indices']) else None)
                for subset in subsets_indices.values()]

        # train_subset = data.Subset(dataset, [2542, 846, 430])
        # valid_subset = data.Subset(dataset, list(range(0, 100)))
        # data_remained = data.Subset(dataset, [101, 102])
        # return [train_subset, valid_subset, data_remained]

    if requires_val and split_rate is None:
        split_rate = [0.8, 0.2]

    ds = COCODataset(folder_path,
                     annotation_path,
                     img_trans if img_trans else transforms.ToTensor(),
                     target_trans)

    if requires_val:
        if data_selectors:
            return selective_split_coco(ds, split_rate, data_selectors)
        else:
            return random_split(ds, split_rate)
    else:
        return ds


class DataSelector:
    """
    DataSelector:
    requires implementation of  '__call__' method that returns BOOL value to select data if satisfy the conditions.
    e.g. select the num of bbox in scene or if this scene is a traffic scene or something.
    """
    def __init__(self, *args):
        pass

    def __call__(self, *args) -> bool:
        pass


class BoxNumSelector(DataSelector):
    def __init__(self, low_thr=0, hi_thr=100):
        super().__init__()
        self.low_thr = low_thr
        self.hi_thr = hi_thr

    def __call__(self, coco_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> bool:
        img_tensor, targets = coco_data
        if self.low_thr <= targets['labels'].shape[0] <= self.hi_thr:
            return True
        else:
            return False


class TrafficScenesSelector(DataSelector):

    coco_traffic_objects = {
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        6: 'bus',
        8: 'truck',
        10: 'traffic light',
        13: 'stop sign',
        14: 'parking meter'
    }

    def __init__(self, num_traffic_object_thr=0):
        super().__init__()
        self.num_traffic_objects_thr = num_traffic_object_thr

    def __call__(self, coco_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> bool:
        img_tensor, targets = coco_data
        traffic_objects_cnt = 0
        for label in targets['labels']:
            # if scene contains 'traffic light' or 'stop sign'
            if label.item() == 10 or label.item() == 13:
                return True
            if label.item() in TrafficScenesSelector.coco_traffic_objects.keys():
                traffic_objects_cnt += 1
        if traffic_objects_cnt >= self.num_traffic_objects_thr:
            return True
        else:
            return False


class ComposeSelector(DataSelector):
    def __init__(self, selectors):
        super().__init__()
        self.selectors = selectors

    def __call__(self, coco_data: Tuple[torch.Tensor, Dict[str, torch.Tensor]]) -> bool:
        for selector in self.selectors:
            if selector(coco_data) is True:
                continue
            else:
                return False

        return True