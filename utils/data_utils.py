from typing import *
import torch
import random
import torch.utils.data as data
from tqdm import tqdm


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

    subsets_indices = {i: {'indices': [], 'length': length} for i, length in enumerate(lengths)}

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
