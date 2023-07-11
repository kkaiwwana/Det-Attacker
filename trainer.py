import torch
import torch.utils.data as data
from typing import List, Dict
from utils.evaluate_utils import log
from utils.utils import default_collate_fn
from utils.visualize_utils import DataVisualizer


class AdvPatchTrainer:
    def __init__(self,
                 net2attack,
                 patch_generator,
                 projector,
                 loss_function,
                 optimizer,
                 scheduler=None,
                 targets_generator=None,
                 device='cuda:0'
                 ):
        self.net2attack = net2attack
        self.patch_generator = patch_generator
        self.projector = projector
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.targets_generator = targets_generator
        self.device = device

    def _model_forward(self, mode: str, dataloader):
        with torch.no_grad():
            if mode == 'classification':
                Y, Y_hat = [], []
                for X, y in dataloader:
                    X, y = X.to(self.device[0]), y.to(self.device[0])
                    X, _, (_, _) = self.projector(X, self.patch_generator(X.shape[0]))
                    Y.append(y)
                    Y_hat.append(self.net2attack(X))
                return torch.concat(Y, dim=0), torch.cat(Y_hat, dim=0)
            elif mode == 'detection':
                # record model's prediction in dataset
                detection: List[Dict] = []
                annotation: List[Dict] = []
                mean_loss: float = 0.0
                pattern = self.patch_generator()
                for images, targets in dataloader:
                    images = list(images)
                    for i in range(len(images)):
                        images[i] = images[i].broadcast_to((3,) + images[i][0].shape).clone()
                        images[i] = images[i].to(self.device[0])
                        images[i], _, (_, _) = self.projector(images[i], pattern)
                    for target in targets:
                        target['boxes'] = target['boxes'].to(self.device[0])
                        target['labels'] = target['labels'].to(self.device[0])
                        target['boxes'][:, 2: 4] += 1e-2
                        annotation.append(target)
                    self.loss_function.eval()
                    self.net2attack.eval()
                    detection.append(self.net2attack(images))
                    self.net2attack.train()
                    toxic_targets = self.targets_generator(targets, self.device[0]) if self.targets_generator else None
                    mean_loss += self.loss_function(self.net2attack(images, targets, toxic_targets)).item()
                    self.loss_function.train()
                return annotation, detection, mean_loss / len(dataloader)

    def _train_epoch(self, epoch, iters_per_image, train_dl, valid_dl, mode, train_watcher, f):
        if mode == 'classification':
            Y, Y_hat = [], []
            for idx, datas in enumerate(train_dl):
                X, y = datas[0].to(self.device[0]), datas[1].to(self.device[0])
                # update pattern
                self.optimizer.zero_grad()
                pattern = self.patch_generator(X.shape[0])
                X, masked_pattern, (_, _) = self.projector(X, pattern)
                y_hat = self.net2attack(X)
                batch_loss = self.loss_function(y_hat, y)

                batch_loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                # statistic data
                Y_hat.append(y_hat.detach())
                Y.append(y)

            valid_Y, valid_Y_hat = None, None
            if valid_dl:
                valid_Y, valid_Y_hat = self._model_forward('classification', train_dl)
            if train_watcher:
                train_watcher(
                    f, {
                        'epoch': epoch,
                        'train_labels': torch.concat(Y, dim=0),
                        'train_predictions': torch.concat(Y_hat, dim=0),
                        'valid_labels': valid_Y,
                        'valid_predictions': valid_Y_hat
                    }
                )
        elif mode == 'detection':
            detection: List[Dict] = []
            annotation: List[Dict] = []
            mean_loss: float = 0.0
            for batch_images, targets in train_dl:
                for iters in range(iters_per_image):
                    images = [image.clone() for image in batch_images]
                    pattern = self.patch_generator()

                    for i in range(len(images)):
                        # some images in COCO are gray-scale image, convert it to RGB image
                        images[i] = images[i].broadcast_to((3,) + images[i][0].shape).clone()
                        # project adv pattern to img
                        images[i] = images[i].to(self.device[0])
                        images[i], _, (_, _) = self.projector(images[i], pattern)
                    for target in targets:
                        target['boxes'] = target['boxes'].to(self.device[0])
                        target['labels'] = target['labels'].to(self.device[0])
                        # some bboxes in COCO have 0 height or width, which are invalid inputs for model
                        # we add a micro positive number to avoid this issue
                        target['boxes'][:, 2: 4] += 1e-2
                        annotation.append(target)
                    self.optimizer.zero_grad()
                    self.net2attack.train()
                    toxic_targets = self.targets_generator(targets, self.device[0]) if self.targets_generator else None
                    losses_dict = self.net2attack(images, targets, toxic_targets)
                    # you should customize loss function by yourself
                    # which returns loss value (required) for patch update and log something if you want.
                    loss = self.loss_function(losses_dict, f)
                    loss.backward()
                    mean_loss += loss.item()
                    self.optimizer.step()
                    torch.cuda.empty_cache()
                    if self.scheduler:
                        self.scheduler.step()
                    self.net2attack.eval()
                    with torch.no_grad():
                        detection.append(self.net2attack(images))

            valid_detection = None
            valid_annotation = None
            valid_mean_loss = None
            if valid_dl:
                valid_annotation, valid_detection, valid_mean_loss = self._model_forward('detection', valid_dl)
            if train_watcher:
                train_watcher(f, {
                                  'epoch': epoch,
                                  'mean_loss': mean_loss / len(train_dl) / iters_per_image,
                                  'annotation': annotation,
                                  'detection': detection,
                                  'valid_mean_loss': valid_mean_loss,
                                  'valid_annotation': valid_annotation,
                                  'valid_detection': valid_detection
                              })

    def train(self,
              mode: str,
              train_ds: torch.utils.data.Dataset,
              batch_size,
              num_epochs,
              iters_per_image=1,
              valid_ds=None,
              log_filepath: str = None,
              num_workers=1,
              train_watcher=None,
              collate_fn=None):
        assert mode == 'classification' or 'detection', \
            f'Mode should be \'classification\' or \'detection\', got \'{mode}\''

        f = open(log_filepath, 'w') if log_filepath else None
        if isinstance(self.device, str):
            self.device = (self.device,)
        if 'cuda' in self.device[0]:
            self.net2attack = self.net2attack.to(self.device[0])
            self.patch_generator = self.patch_generator.to(self.device[0])
            if len(self.device) > 1:
                self.net2attack = torch.nn.DataParallel(self.net2attack, device_ids=self.device)
                self.patch_generator = self.patch_generator.to(self.device[0])

        if mode == 'detection' and collate_fn is None:
            collate_fn = default_collate_fn
        else:
            collate_fn = None
        train_loader = data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn)
        valid_loader = data.DataLoader(valid_ds,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn) if valid_ds else None
        import time
        start_time = time.time()

        for epoch in range(num_epochs):
            self._train_epoch(epoch, iters_per_image, train_loader, valid_loader, mode, train_watcher, f)

            if epoch == 0:
                log(f'The training procedure will be completed at about '
                    f'{time.asctime(time.localtime(time.time() + (time.time() - start_time) * (num_epochs - 1)))}\n',
                    f=f)
            log(f'epoch: {epoch + 1}  {time.asctime(time.localtime(time.time()))}', f=f)

        log('\n\n------FINISHED TRAINING------\n\n', f=f)

    def save_model(self, filename, filepath='./'):
        torch.save(self.patch_generator, filepath + filename)
