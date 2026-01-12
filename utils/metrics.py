import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    device = y_true.device if isinstance(y_true, torch.Tensor) else 'cpu'
    
    # 转换为tensor并确保在GPU上
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.from_numpy(y_true).to(device)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.from_numpy(y_pred).to(device)
    
    # 处理预测输出
    if y_pred.dim() == 4 and y_pred.shape[1] > 1:  # [B, C, H, W]
        y_pred = y_pred.argmax(1)  # 在GPU上argmax
    elif y_pred.dim() == 4 and y_pred.shape[1] == 1:  # [B, 1, H, W]
        y_pred = (y_pred > 0.5).squeeze(1).long()
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    indices = num_classes * y_true_flat + y_pred_flat
    cm = torch.bincount(indices, minlength=num_classes**2)
    cm = cm.reshape(num_classes, num_classes)
    
    return cm.cpu().numpy()

def calculate_iou(confusion_matrix):
    smooth = 1e-10
    
    true_positive = np.diag(confusion_matrix)
    ground_truth_sum = np.sum(confusion_matrix, axis=1)
    predicted_sum = np.sum(confusion_matrix, axis=0)
    
    union = ground_truth_sum + predicted_sum - true_positive
    iou_per_class = (true_positive + smooth) / (union + smooth)
    
    valid_iou = iou_per_class[~np.isnan(iou_per_class)]
    mean_iou = np.mean(valid_iou) if len(valid_iou) > 0 else 0.0

    return iou_per_class, mean_iou

def calculate_accuracy(confusion_matrix):
    true_predictions = np.sum(np.diag(confusion_matrix))
    total_pixels = np.sum(confusion_matrix)
    
    if total_pixels == 0:
        return 0.0
    
    return true_predictions / total_pixels

def calculate_f1_score(confusion_matrix):
    smooth = 1e-10
    
    true_positive = np.diag(confusion_matrix)
    ground_truth_sum = np.sum(confusion_matrix, axis=1)
    predicted_sum = np.sum(confusion_matrix, axis=0)
    
    precision = (true_positive + smooth) / (predicted_sum + smooth)
    recall = (true_positive + smooth) / (ground_truth_sum + smooth)
    
    f1_per_class = 2 * (precision * recall) / (precision + recall + smooth)
    valid_f1 = f1_per_class[~np.isnan(f1_per_class)]
    mean_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0.0
    
    return f1_per_class, mean_f1

def evaluate_prediction(y_true, y_pred, num_classes=2):
    device = y_true.device if isinstance(y_true, torch.Tensor) else 'cpu'
    
    # 确保数据在GPU上
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.from_numpy(y_true).to(device)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.from_numpy(y_pred).to(device)
    
    if y_pred.dim() == 4 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(1)
    elif y_pred.dim() == 4 and y_pred.shape[1] == 1:
        y_pred = (y_pred > 0.5).squeeze(1).long()

    smooth = 1e-10
    
    correct = (y_pred == y_true).float()
    accuracy = correct.mean().item()

    ious = []
    f1s = []
    
    for cls in range(num_classes):
        pred_cls = (y_pred == cls).float()
        true_cls = (y_true == cls).float()
        
        # IoU
        intersection = (pred_cls * true_cls).sum()
        union = pred_cls.sum() + true_cls.sum() - intersection
        
        if union > 0:
            iou = (intersection / union).item()
            ious.append(iou)
            
            # F1
            precision = intersection / (pred_cls.sum() + smooth)
            recall = intersection / (true_cls.sum() + smooth)
            f1 = (2 * precision * recall / (precision + recall + smooth)).item()
            f1s.append(f1)
    
    mean_iou = np.mean(ious) if ious else 0.0
    mean_f1 = np.mean(f1s) if f1s else 0.0

    cm = calculate_confusion_matrix(y_true, y_pred, num_classes)

    return {
        'confusion_matrix': cm,
        'iou_per_class': np.array(ious + [0.0] * (num_classes - len(ious))),
        'mean_iou': mean_iou,
        'accuracy': accuracy,
        'f1_per_class': np.array(f1s + [0.0] * (num_classes - len(f1s))),
        'mean_f1': mean_f1
    }

def evaluate_prediction_ultrafast(y_true, y_pred, num_classes=2):
    device = y_true.device if isinstance(y_true, torch.Tensor) else 'cpu'
    
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.from_numpy(y_true).to(device)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.from_numpy(y_pred).to(device)
    
    if y_pred.dim() == 4 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(1)
    elif y_pred.dim() == 4 and y_pred.shape[1] == 1:
        y_pred = (y_pred > 0.5).squeeze(1).long()
    
    # 只计算必要指标
    correct = (y_pred == y_true).float()
    accuracy = correct.mean().item()
    
    if num_classes == 2:
        fg_pred = (y_pred == 1).float()
        fg_true = (y_true == 1).float()

        intersection = (fg_pred * fg_true).sum()
        union = fg_pred.sum() + fg_true.sum() - intersection

        mean_iou = (intersection / union).item() if union > 0 else 0.0
    else:
        ious = []
        for cls in range(num_classes):
            pred_cls = (y_pred == cls).float()
            true_cls = (y_true == cls).float()

            intersection = (pred_cls * true_cls).sum()
            union = pred_cls.sum() + true_cls.sum() - intersection

            if union > 0:
                ious.append((intersection / union).item())

        mean_iou = np.mean(ious) if ious else 0.0

    return {
        'mean_iou': mean_iou,
        'accuracy': accuracy,
        'mean_f1': mean_iou
    }