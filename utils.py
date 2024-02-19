import numpy as np
import torch
from scipy.ndimage import label
from torch import tensor, nn
from torch.nn import init
import h5py
import random
import gc
from math import ceil

mean_global = 1.8726
std_global = 1.0707

DATASET_LENGTH = 369


# def load_volume_old(path, volume):
#     h5file = h5py.File(path+f"volume_{volume}_slice_0.h5")
#     image = np.array(h5file["image"])[:, None]
#     mask = np.array(h5file["mask"])[:, None]
#     for i in range(1, 155):
#         h5file = h5py.File(path+f"volume_{volume}_slice_{i}.h5")
#         image = np.concatenate((image, np.array(h5file["image"])[:, None]), axis=1)
#         mask = np.concatenate((mask, np.array(h5file["mask"])[:, None]), axis=1)
#     return tensor(image).permute(3, 0, 1, 2), tensor(mask).permute(3, 0, 1, 2)


def load_volume(path, volume):
    images = []
    masks = []
    for i in range(155):
        file = h5py.File(path + f"volume_{volume}_slice_{i}.h5")
        images.append(np.array(file["image"]))
        masks.append(np.array(file["mask"]))
    image = np.stack(images, axis=1)
    mask = np.stack(masks, axis=1)
    return tensor(image).permute(3, 0, 1, 2), tensor(mask).permute(3, 0, 1, 2)


def postprocessing(out):
    """
    :param out: one hot encoded pred(C, D, H, W)
    :return: postprocessed prediction
    """
    clusters = out[1:].sum(0)  # get tumor area
    structure = torch.ones((3, 3, 3), dtype=torch.int)
    labeled, num_clusters = label(clusters, structure)
    if type(labeled) is not tensor:
        labeled = tensor(labeled)

    for i in range(num_clusters):
        where_cluster = (labeled == (i + 1))
        if where_cluster.float().sum() < 100:
            labeled[where_cluster] = 0

    new_pred = torch.where(labeled > 0, out, torch.ones(out.shape) * torch.tensor([1, 0, 0, 0]).reshape(4, 1, 1, 1))
    return new_pred


def normalize(inp, epsilon=1e-7, mean=None, std=None):
    if mean is None or std is None:
        nz = inp[inp > 0]
        if mean is None:
            mean = nz.mean()
        if std is None:
            std = nz.std() + epsilon
    out = inp - mean
    out /= std
    out = out * ((inp > 0).float())
    return out


def init_weights_kaiming(m, leaky=0.):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, leaky)


def noop(*args):
    if len(args) > 1:
        return args
    return args[0]


def sampling(x, y, size):
    """
    :param x: input tensor shape C, D, H, W
    :param y: mask tensor shape C, D, H, W
    :param size: size cubed of selected patch
    :return: patch
    """
    selectable = y[1:].sum(0)
    start, stop = ceil(size / 2) - 1, -(size // 2)
    selectable = selectable[start:stop, start:stop, start:stop]
    if size <= 64 and random.random() <= 0.5:
        cond = 0
    else:
        cond = 1
    a, b, c = torch.where(selectable == cond)
    if len(a) == 0:
        a, b, c = torch.where(selectable == (1 - cond))
    i = random.randint(0, len(a) - 1)
    ai, bi, ci = a[i], b[i], c[i]
    patch_x = x[:, ai: ai - stop + start + 1, bi: bi - stop + start + 1, ci: ci - stop + start + 1]
    patch_y = y[:, ai: ai - stop + start + 1, bi: bi - stop + start + 1, ci: ci - stop + start + 1]
    return patch_x, patch_y


def generalized_dice_loss(pred: tensor, target: tensor, epsilon=1e-6, weight_type="simple"):
    """
    :param pred: (N, C, D, H, W)
    :param target: (N, C, D, H, W)
    :param epsilon:
    :param weight_type: "simple" or "square"
    :return:
    """
    pred = pred.permute(1, 0, 2, 3, 4).contiguous()
    target = target.permute(1, 0, 2, 3, 4).contiguous()
    c = target.shape[0]

    pred = pred.view(c, -1)
    target = target.view(c, -1)

    w = target.sum(1)
    if weight_type == "square":
        w = w * w
    w = 1 / w.clamp(min=epsilon)
    w.requires_grad = False

    intersection = 2 * (pred * target).sum(1) * w
    denominator = (pred + target).sum(1) * w
    denominator = denominator.clamp(min=epsilon)
    return 1 - (intersection.sum() / denominator.sum())


class RandomTransformation(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x, y):
        if random.random() <= self.p:
            return self.functionality(x, y)
        return x, y

    def functionality(self, x, y):
        return x, y


class RandomFlip(RandomTransformation):
    def __init__(self, p, axis):
        super().__init__(p)
        self.axis = axis

    def functionality(self, x, y):
        return x.flip(self.axis), y.flip(self.axis)


class RandomRotate(RandomTransformation):
    def __init__(self, p, axis):
        super().__init__(p)
        self.axis = axis

    def functionality(self, x, y):
        k = random.randint(1, 3)
        return x.rot90(k, self.axis), y.rot90(k, self.axis)


class RandomIntensityScale(nn.Module):
    def __init__(self, scale_min=0.9, scale_max=1.1):
        super().__init__()
        self.min = scale_min
        self.max = scale_max

    def forward(self, x, y):
        return x * random.uniform(self.min, self.max), y


class RandomIntensityShift(nn.Module):
    def __init__(self, shift_min=-0.1, shift_max=0.1):
        super().__init__()
        self.min = shift_min
        self.max = shift_max

    def forward(self, x, y):
        mask = y.sum(0)
        for i, channel in enumerate(x):
            shift = random.uniform(self.min, self.max)
            std = torch.std(channel[mask == 1])
            x[i] = channel + std * shift
        return x, y


class EmptyDataset:
    def __len__(self):
        return 0

    def __getitem__(self, item):
        return item


class BrainDataset:
    def __init__(self, func, start=0, sampling_func=noop, percent: float = 1., device="cpu"):
        self.function = func
        self.x = None
        self.y = None
        self.start = start
        self.sampling_func = sampling_func
        self.device = device
        self.percent = percent

    def __len__(self):
        return round(DATASET_LENGTH * self.percent)

    def __getitem__(self, item):
        print(item)
        del self.x, self.y
        if self.device == "cpu":
            gc.collect()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        self.x, self.y = self.function(item + self.start)
        self.x, self.y = self.sampling_func(self.x, self.y)
        self.x, self.y = self.x.float(), self.y.float()
        return self.x.to(self.device), self.y.to(self.device)
