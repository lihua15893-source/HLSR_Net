import time

from PIL import Image
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
import copy
import math
from functools import partial
import pickle
import random
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")

def row_normalize(adj_mat):
    Q = torch.sum(adj_mat, dim=-1).float()
    sQ = torch.rsqrt(Q)
    sQ = torch.diag(sQ)
    return torch.mm(sQ, torch.mm(adj_mat, sQ))


def normalize_adjacency(adj_mat):
    assert adj_mat.shape[0] == adj_mat.shape[1]
    adj_mat += torch.eye(adj_mat.shape[0], device=adj_mat.device)
    adj_mat = adj_mat.float()
    norm_adj_mat = row_normalize(adj_mat)
    return norm_adj_mat


def get_pickle_data(pickle_file_path):
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)
    return data
    
def read_image(path, isLabel=False):
    if not isLabel:
        image = Image.open(path).convert('RGB')
    else:
        image = Image.open(path).convert('L')
    return image
def Resize_Image(image, size, mode='image'):

    w, h = image.size
    scale = min(size / w, size / h)
    nw = int(w * scale)
    nh = int(h * scale)
    if mode == 'image':
        image = image.resize((nw, nh), Image.BICUBIC).convert('RGB')
    elif mode == 'label':
        image = image.resize((nw, nh), Image.NEAREST).convert('L')
    else:
        raise TypeError('The mode of image is incorrect!')
    return image

def keep_image_size(path, size = 256, val_mode = True ):
    if val_mode:
        img = Image.open(path).convert('L')
        w, h = img.size
        scale = min(size / w, size / h)
        nw = int(w * scale)
        nh = int(h * scale)
        img = img.resize((nw, nh), Image.NEAREST)
        mask = Image.new('L', (size, size), 0)

    else:
        img = Image.open(path).convert('RGB')
        w, h = img.size
        scale = min(size / w, size / h)
        nw = int(w * scale)
        nh = int(h * scale)
        img = img.resize((nw, nh), Image.BICUBIC)
        mask = Image.new('RGB', (size, size), (0, 0, 0))

    mask.paste(img, ((size-nw)//2,(size-nh)//2))

    return mask

def label_norm(label):
    label = np.array(label)
    label[label >= 1] = 1
    return label

def flip_and_rotataion(image, label):
    dice = rand()
    if 0.5 < dice <0.9:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    elif dice >= 0.9:
        image = image.transpose(Image.ROTATE_90)
        label = label.transpose(Image.ROTATE_90)
    return image, label

def rand():
    dice = np.random.rand()
    return dice

def normal_image(image):
    image /= 255
    return image

def Compute_Score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    # nt, ht, wt, ct = target.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1)
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def CalculatingMIoU(label_img, pre_img, num_class):
    # true_img: b h w 形状, pre_img: b c h w
    label_img = copy.deepcopy(label_img)
    pre_img = copy.deepcopy(pre_img)

    confusion_matrixs = np.zeros((num_class,num_class), np.uint8)
    true_img = np.array(label_img.cpu())    # 将tensor拷贝到cpu
    pre_img = np.array(pre_img.cpu()).argmax(1)   # 去掉c维
    meanIoU = 0
    acc = 0
    F1 = 0
    for i in range(true_img.shape[0]):
        true = true_img[i,:,:]
        predict = pre_img[i,:,:]
        single_confusion_matrix = confusion_matrix(true.flatten(), predict.flatten(), labels=[i for i in range(num_class)])
        confusion_matrixs = confusion_matrixs + single_confusion_matrix
        meanIoU += Calculate_MIoU(confusion_matrixs)
        acc += Calculate_Accuracy(confusion_matrixs)
        F1 += CalculatingF1Score(confusion_matrixs)
    return meanIoU, acc, F1

def Calculate_MIoU(confusion_matrixs):
    intersection = np.diag(confusion_matrixs)
    union = np.sum(confusion_matrixs, axis=1) + np.sum(confusion_matrixs, axis=0) - np.diag(confusion_matrixs)
    union[union<=0] = 1e-10
    IoU = intersection / union
    MeanIoU = np.nanmean(IoU)
    return MeanIoU

def Calculate_Accuracy(confusion_matrixs):
    return np.diag(confusion_matrixs).sum() / confusion_matrixs.sum()

def CalculatingF1Score(confusion_matrixs):
    Precision = np.diag(confusion_matrixs) / (confusion_matrixs.sum(axis=0)+1e-10)
    Recall = np.diag(confusion_matrixs) / (confusion_matrixs.sum(axis=1) +1e-10)
    F1Score = (2 * Precision * Recall) / (Precision + Recall +1e-10)

    return F1Score

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    path = '../data-432/train/image/data0124.PNG'
    image = Image.open(path)
    image = Resize_Image(image,512,'label')
    img = np.array(image)
    print(img.shape)
    image.show()

