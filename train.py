import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import logging
from datetime import datetime

from UNIVERSAL_CONFIG import UNIVERSAL_CONFIG
from utils.trainer_utils import (
    get_optimizer, get_scheduler, save_checkpoint, load_checkpoint,
    get_lr, get_loss_function, train_one_epoch, validate
)
from torch.utils.data import DataLoader
from utils.metrics import evaluate_prediction
from dataset import my_Dataset
from HLSR_Net import HLSR

def cleanup_models(checkpoint_dir, keep_best=3):
    """Keep best 3 models, delete others"""
    import glob

    model_files = glob.glob(os.path.join(checkpoint_dir, 'best_model_*.pth'))

    if len(model_files) <= keep_best:
        return

    def extract_miou(filename):
        try:
            miou_str = filename.split('best_model_')[1].split('.pth')[0]
            return float(miou_str)
        except:
            return 0.0

    model_files.sort(key=extract_miou, reverse=True)

    files_to_delete = model_files[keep_best:]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logging.info(f"Cleaned old model: {os.path.basename(file_path)}")
        except:
            pass


def main(args):
    config = UNIVERSAL_CONFIG

    use_pretrained = args.pretrained if args.pretrained is not None else config.USE_PRETRAINED

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.loss:
        config.LOSS_TYPE = args.loss

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(
        config.OUTPUT_DIR,
        f"HLSR_{timestamp}"
    ))
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "training.log")
    log_txt_path = os.path.join(run_dir, "console_output.txt")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    text_handler = logging.FileHandler(log_txt_path)
    text_handler.setLevel(logging.INFO)
    text_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(text_handler)

    device = torch.device(config.DEVICE)

    logging.info(f"Batch size: {config.BATCH_SIZE}")
    logging.info(f"Epochs: {config.EPOCHS}")
    logging.info(f"Learning rate: {config.LEARNING_RATE}")
    logging.info(f"Device: {device}")
    train_dataset = my_Dataset(size=config.IMAGE_SIZE, mode='train', num_class=config.NUM_CLASSES, enhance=config.USE_AUGMENTATION)
    val_dataset = my_Dataset(size=config.IMAGE_SIZE, mode='val', num_class=config.NUM_CLASSES, enhance=False)

    def worker_init_fn(worker_id):
        pass

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        worker_init_fn=worker_init_fn
    )

    logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = HLSR(num_classes=config.NUM_CLASSES, pretrained=use_pretrained)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total params: {total_params:,}")
    logging.info(f"Trainable params: {trainable_params:,}")

    with torch.no_grad():
        test_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
        test_output = model(test_input)
        logging.info(f"Model test passed: {test_input.shape} -> {test_output.shape}")
        del test_input, test_output
    criterion = get_loss_function(
        config.LOSS_TYPE,
        config.NUM_CLASSES,
        class_weights=None,
        dice_weight=config.DICE_WEIGHT,
        ce_weight=config.CE_WEIGHT
    )

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    train_losses = []
    val_losses = []
    val_mious = []
    val_accuracies = []

    best_miou = 0
    logging.info("Starting training...")

    for epoch in range(config.EPOCHS):
        logging.info(f"Epoch {epoch+1}/{config.EPOCHS}")
        start_time = time.time()
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=config.EPOCHS,
        )
        train_time = time.time() - start_time

        if (epoch + 1) % config.VAL_FREQ == 0:
            start_time = time.time()
            val_loss, miou, accuracy, f1 = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=config.NUM_CLASSES,
                evaluate_func=evaluate_prediction
            )
            val_time = time.time() - start_time
            if config.LR_SCHEDULER == 'reduce':
                scheduler.step(miou)
            else:
                scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_mious.append(miou)
            val_accuracies.append(accuracy)
            logging.info(f"Train Loss: {train_loss:.4f} ({train_time:.2f}s)")
            logging.info(f"Val Loss: {val_loss:.4f} ({val_time:.2f}s)")
            logging.info(f"mIoU: {miou:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
            logging.info(f"LR: {get_lr(optimizer):.6f}")
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_{best_miou:.4f}.pth'))
                logging.info(f"Saved best model, mIoU: {best_miou:.4f}")
                cleanup_models(checkpoint_dir, keep_best=3)
            if epoch == config.EPOCHS - 1:
                torch.save(model.state_dict(),
                          os.path.join(checkpoint_dir, f"final_epoch_{epoch + 1}_miou_{miou:.4f}.pth"))
                logging.info(f"Saved final model: final_epoch_{epoch + 1}_miou_{miou:.4f}.pth")
        else:
            if config.LR_SCHEDULER != 'reduce':
                scheduler.step()

            logging.info(f"Train Loss: {train_loss:.4f} ({train_time:.2f}s)")
            logging.info(f"LR: {get_lr(optimizer):.6f}")

        if config.EARLY_STOPPING and len(val_mious) > config.PATIENCE:
            if max(val_mious[-config.PATIENCE:]) < best_miou:
                logging.info(f"Early stopping at epoch {epoch+1}, best mIoU: {best_miou:.4f}")
                break

    logging.info("Training completed!")
    logging.info(f"Best mIoU: {best_miou:.4f}")
    best_model_path = os.path.join(checkpoint_dir, f'best_model_{best_miou:.4f}.pth')
    logging.info(f"Best model path: {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLSR segmentation training script")

    parser.add_argument("--pretrained", type=lambda x: x.lower() in ['true', '1', 'yes'],
                       default=None, help="Use pretrained weights")

    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--loss", type=str, choices=['ce'], help="Loss function type")
    parser.add_argument("--output_dir", type=str, help="Output directory")

    args = parser.parse_args()
    main(args)
