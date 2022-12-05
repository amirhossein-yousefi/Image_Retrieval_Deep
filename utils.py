import os
import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(root_dir, split):
    filename = os.path.join(root_dir, 'test_32x32.mat')
    if (split.startswith('train') or split.startswith('unlabelled')):
        filename = os.path.join(root_dir, 'train_32x32.mat')
    elif (split.startswith('test')):
        filename = os.path.join(root_dir, 'test_32x32.mat')

    loaded_mat = sio.loadmat(filename)
    imgs = (loaded_mat['X']).astype(np.float32)
    labels = loaded_mat['y'].astype(np.int64).squeeze()
    if (split == 'train_29_task2'):
        imgs_idx_01 = np.logical_or(labels == 10, labels == 1)
        imgs_idx_29 = np.where(np.logical_not(imgs_idx_01))
        imgs = imgs[:, :, :, imgs_idx_29]
        labels = labels[imgs_idx_29]
    elif (split == 'test_01_task2' or split == 'train_01_task2'):
        imgs_idx_01 = np.where(np.logical_or(labels == 10, labels == 1))[0]
        if (split == 'train_01_task2'):
            imgs_idx_01 = imgs_idx_01[0:200]
        else:
            imgs_idx_01 = imgs_idx_01[200::]
        imgs = imgs[:, :, :, imgs_idx_01]
        labels = labels[imgs_idx_01]
    if (split == 'test_task3'):
        N = 50
        imgs = imgs[:, :, :, 0:N]
        labels = labels[0:N]
    print('Loaded SVHN split: {split}'.format(split=split))
    print('-------------------------------------')
    print('Images Size: ', imgs.shape[0:-1])
    print('Split Number of Images:', imgs.shape[-1])
    print('Split Labels Array Size:', labels.shape)
    print('Possible Labels: ', np.unique(labels))
    return imgs, labels


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_app2_2_trans.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def visualize_image(images, labels, index):
    img = images[:, :, :, index]
    label = labels[index]
    plt.imshow((img))
    plt.text(1, 3, 'Label: {label}'.format(label=label), c='red', fontsize=20,
             bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.axis('off')
    plt.show()
