import torch
from typing import *
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.utils.data as data
from tqdm import tqdm
from visualize_utils import DataVisualizer


def default_collate_fn(batch):
    return tuple(zip(*batch))


class AdvDetectionMetrics:
    # TODO test this class later.
    """
    Compute metrics on detection task for adversarial patch
    """
    def __init__(self, model, pattern_projector, patch: torch.Tensor or Callable, img_trans=None, target_trans=None):
        self.model = model
        self.projector = pattern_projector
        self.patch = patch
        self.image_trans = img_trans
        self.target_trans = target_trans
        self.metrics = {}

    @staticmethod
    def _boxes_outside_area(
            targets: Tuple[Dict[str, torch.Tensor]],
            areas: torch.Tensor or List[torch.Tensor],
            device: str
    ):
        if isinstance(areas, torch.Tensor):
            if len(areas.shape) == 1:
                areas = areas.unsqueeze(dim=0).broadcast_to((len(targets), -1))
        # return idx of targets not in suppress region
        if targets[0]['boxes'].shape[0] == 0:
            return torch.tensor([])
        not_in_area_idxes = []
        _fl = torch.tensor([True, True, False, False], device=device)

        for target, area in zip(targets, areas):
            if len(target['boxes']) == 0:
                continue
            not_in_area_idxes.append(((target['boxes'] >= area.to(device)) != _fl).any(dim=1))

        return not_in_area_idxes

    def compute(self, dataset, test_clear_imgs=False, batch_size=16, device='cuda', num_workers=1, collate_fn=None):

        num_boxes_with_patch = 0  # number of boxes predicted in images with patch
        num_boxes_unsuppressed = 0  # number of boxes unsuppressed in images with patch
        mAP_with_patch = MeanAveragePrecision()  # mAP values

        num_boxes_clean_images = 0  # number of boxes predicted in clean image
        num_boxes_clean_images_outside_patch = 0  # number of boxes predicted in clean image and outside the patch area
        mAP_clean_image = MeanAveragePrecision()  # mAP values in clean images

        self.model.to(device)
        self.model.eval()
        if collate_fn:
            dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
        else:
            dl = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                 collate_fn=default_collate_fn)

        with tqdm(total=len(dataset)) as pbar:
            pbar.set_description('Processing')
            patch_areas = []
            patch = self.patch().to(device) if callable(self.patch) else self.patch.to(device)
            assert isinstance(patch, torch.Tensor) and len(patch.shape) == 3, \
                'Invalid patch. Arg \'patch\' passed should be either a tensor directly or a callable tensor generator'
            patch_size = patch.shape[1:]
            with torch.no_grad():
                for imgs, targets in dl:
                    # move images to device specified and synchronize channels
                    imgs = list(imgs)
                    for i in range(len(imgs)):
                        imgs[i] = imgs[i].broadcast_to((3,) + imgs[i][0].shape).clone()
                        imgs[i] = imgs[i].to(device)
                    # project adv patch to a copy of origin images
                    imgs_with_patch = []
                    for img in imgs:
                        img_with_patch, _, (posi_x, posi_y) = self.projector(img.clone(), patch)
                        patch_areas.append(
                            torch.tensor([posi_x, posi_y, patch_size[0], patch_size[1]], dtype=torch.float)
                        )
                        imgs_with_patch.append(img_with_patch)
                    # model forward, get predictions
                    preds_with_patch = self.model(imgs_with_patch)
                    mAP_with_patch.update(preds_with_patch, targets)
                    result_with_patch = self._boxes_outside_area(preds_with_patch, patch_areas, device)
                    # accumulate statistic datas

                    num_boxes_with_patch += sum([idx_tensor.shape[0] for idx_tensor in result_with_patch])
                    num_boxes_unsuppressed += sum([idx_tensor.sum().item() for idx_tensor in result_with_patch])
                    # compute other stuffs when testing in clean image
                    if test_clear_imgs:
                        preds = self.model(imgs)
                        mAP_clean_image.update(preds, targets)
                        result = self._boxes_outside_area(preds, patch_areas, device)
                        num_boxes_clean_images += sum([idx_tensor.shape[0] for idx_tensor in result])
                        num_boxes_clean_images_outside_patch += sum([idx_tensor.sum().item() for idx_tensor in result])

                    pbar.update(len(imgs))

        print('Computing metrics. It (especially \'mAP\' items) may take few minutes.')
        mAP_with_patch.to(device)
        mAP_clean_image.to(device)
        if test_clear_imgs:
            self.metrics = {
                'Statistic_Info': {
                    'num_boxes_clean_images': num_boxes_clean_images,
                    'num_boxes_clean_images_outside_patch': num_boxes_clean_images_outside_patch,
                    'num_boxes_with_patch': num_boxes_with_patch,
                    'num_boxes_unsuppressed': num_boxes_unsuppressed
                },
                'Average_Boxes_Number_Increase': (num_boxes_with_patch - num_boxes_unsuppressed) / len(dataset),
                'Boxes_Suppression_Rate': 1 - num_boxes_unsuppressed / num_boxes_clean_images_outside_patch,
                'mAPs_clean_image': mAP_clean_image.compute(),
                'mAPs_with_patch': mAP_with_patch.compute()
            }
        else:

            self.metrics = {
                'Statistic_Info': {
                    'num_boxes_with_patch': num_boxes_with_patch,
                    'num_boxes_unsuppressed': num_boxes_unsuppressed
                },
                'Average_Boxes_Number_Increase': (num_boxes_with_patch - num_boxes_unsuppressed) / len(dataset),
                'Average_Boxes_Number_Unsuppressed': num_boxes_unsuppressed / len(dataset),
                'mAPs_with_patch': mAP_with_patch.compute()
            }
        return self.metrics


class TrainWatcher:
    def __init__(self):
        self.visualizer = DataVisualizer()
        self.train_datas = []
        return

    def __call__(self, f, train_data):
        self.visualizer.record(**{
            'epoch': train_data['epoch'],
            'train_loss': train_data['mean_loss'],
            'valid_loss': train_data['valid_mean_loss']
        })
        self.train_datas.append(train_data)

    def save_data(self, filepath, filename):
        torch.save(self.train_datas, filepath + filename)

    def save_fig(self, filepath, filename):
        plt = self.visualizer.visualize(x_axis_key='epoch', y_axis_keys=[['train_loss', 'valid_loss']])
        plt.savefig(filepath + filename, format='svg')


def simple_watcher(epoch, Y, Y_hat, valid_Y, valid_Y_hat, f):
    # on classification, compute ASR (Attack Success Rate)
    train_acc = (Y_hat.argmax(dim=1) != Y).sum().item() / Y_hat.shape[0]
    valid_acc = (valid_Y_hat.argmax(dim=1) != valid_Y).sum().item() / Y_hat.shape[0]
    log(f'epoch: {epoch} train ASR: {100 * train_acc:.1f}% valid ASR: {100 * valid_acc:.1f}%', f=f)


def log(*args, f=None):
    # simple log, print info to both console and file
    print(args)
    if f:
        for item in args:
            f.write(str(item))
        f.write('\n')
