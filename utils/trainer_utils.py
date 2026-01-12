import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from tqdm import tqdm
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(config, model):
    if config.OPTIMIZER.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.OPTIMIZER}")

def get_scheduler(config, optimizer):
    if config.LR_SCHEDULER.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA
        )
    elif config.LR_SCHEDULER.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS,
            eta_min=config.LR_MIN
        )
    elif config.LR_SCHEDULER.lower() == 'reduce':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.LR_GAMMA,
            patience=5,
            min_lr=config.LR_MIN
        )
    else:
        raise ValueError(f"Unsupported LR scheduler type: {config.LR_SCHEDULER}")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir, is_best=False, filename=None):
    if filename is None:
        filename = f'checkpoint_epoch_{epoch + 1}.pth'

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_metrics': val_metrics
    }

    torch.save(checkpoint, checkpoint_path)
    logging.info(f'Checkpoint saved to {checkpoint_path}')

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        logging.info(f'Best model saved to {best_path}')

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return 0, None

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    logging.info(f'Checkpoint loaded from {checkpoint_path} (epoch {checkpoint["epoch"]})')

    return checkpoint['epoch'], checkpoint.get('val_metrics', None)

def plot_metrics(train_losses, val_losses, val_mious, val_accuracies, save_dir):
    pass

def get_lr_scheduler_func(lr_decay_type, init_lr, min_lr, epochs):
    def cosine_scheduler(epoch):
        return min_lr + 0.5 * (init_lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs))
    
    def step_scheduler(epoch):
        # 可以根据需要实现阶梯式下降
        if epoch < epochs * 0.3:
            return init_lr
        elif epoch < epochs * 0.6:
            return init_lr * 0.1
        else:
            return init_lr * 0.01
    
    if lr_decay_type.lower() == 'cos':
        return cosine_scheduler
    elif lr_decay_type.lower() == 'step':
        return step_scheduler
    else:
        # 默认返回常数学习率
        return lambda epoch: init_lr

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(net, init_type='normal', init_gain=0.02):
    """初始化网络权重"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    logging.info(f'Initializing weights with {init_type} method')
    net.apply(init_func)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, scaler=None, grad_accum_steps=1, log_freq=100):
    model.train()

    # 初始化跟踪变量
    epoch_loss = 0
    step = 0

    with tqdm(total=len(train_loader), desc=f'Train Epoch {epoch + 1}/{total_epochs}') as pbar:
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss = loss / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps
            step += 1

            pbar.set_postfix({'loss': epoch_loss / step, 'lr': get_lr(optimizer)})
            pbar.update(1)

            if (i + 1) % log_freq == 0:
                logging.info(
                    f'Epoch {epoch + 1}, Step {i + 1}/{len(train_loader)}, Loss: {epoch_loss / step:.4f}, LR: {get_lr(optimizer):.6f}')

    return epoch_loss / step

def validate(model, val_loader, criterion, device, num_classes, evaluate_func):
    model.eval()

    val_loss = 0
    all_metrics = []

    with torch.no_grad():
        with tqdm(total=len(val_loader), desc='Validation') as pbar:
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                metrics = evaluate_func(labels, outputs, num_classes)
                all_metrics.append(metrics)

                pbar.set_postfix({'loss': val_loss / (pbar.n + 1), 'miou': metrics['mean_iou']})
                pbar.update(1)

    val_loss /= len(val_loader)

    mean_iou = np.mean([m['mean_iou'] for m in all_metrics])
    accuracy = np.mean([m['accuracy'] for m in all_metrics])
    mean_f1 = np.mean([m['mean_f1'] for m in all_metrics])

    logging.info(
        f'Validation Results: Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}, Accuracy: {accuracy:.4f}, Mean F1: {mean_f1:.4f}')

    return val_loss, mean_iou, accuracy, mean_f1

def load_pretrained_weights(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        logging.warning(f"Pretrained weights file not found: {pretrained_path}")
        return False

    try:
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')

        if isinstance(pretrained_dict, dict) and 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        if hasattr(model, 'load_pretrained_weights'):
            success = model.load_pretrained_weights(pretrained_path)
            if success:
                logging.info(f"Loaded pretrained weights using custom method: {pretrained_path}")
                return True

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict and model_dict[k].shape == v.shape}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        logging.info(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained weights")
        return True
    except Exception as e:
        logging.error(f"Failed to load pretrained weights: {e}")
        return False

def get_loss_function(loss_type, num_classes, class_weights=None, **kwargs):
    loss_type = loss_type.lower()

    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")