# ============================
# train.py
# ============================
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.data import decollate_batch

from model_unetR import model
from transform_train_test_images_with_augmentations import Dataset, PairedTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
model.to(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MONAI U-Net.")
    parser.add_argument("--dataset_path", type=str, default="train_data_transformed.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()

def combined_dice_bce(pred, target):
    dice = DiceLoss(sigmoid=True)(pred, target)
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean()
    return dice + bce

def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, "saved_images")
    model_save_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    dataset = torch.load(args.dataset_path, weights_only=False)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete()])

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []
    counter = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for step, batch_data in enumerate(train_loader):
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            loss = combined_dice_bce(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_inputs)
                    loss = combined_dice_bce(val_outputs, val_labels)
                    val_loss += loss.item()

                    val_outputs_bin = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels_bin = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs_bin, y=val_labels_bin)

                val_loss /= len(val_loader)
                val_loss_values.append(val_loss)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_save_dir, "best_metric_model.pth"))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= args.patience:
                        print(f"[INFO] Early stopping at epoch {epoch+1}")
                        break

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Loss")
    plt.plot(epoch_loss_values, label='Train')
    plt.plot([i*args.val_interval for i in range(len(val_loss_values))], val_loss_values, label='Val')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Validation Dice")
    plt.plot([i*args.val_interval for i in range(len(metric_values))], metric_values)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_image, val_label = val_data[0].to(device), val_data[1].to(device)
            val_output = model(val_image)
            val_output_bin = torch.sigmoid(val_output) > 0.5
            fp = (val_output_bin == 1) & (val_label == 0)
            fn = (val_output_bin == 0) & (val_label == 1)

            img = val_image[0].permute(1, 2, 0).cpu().numpy()
            gt = val_label[0, 0].cpu().numpy()
            pred = val_output_bin[0, 0].cpu().numpy()
            fp_img = fp[0, 0].cpu().numpy()
            fn_img = fn[0, 0].cpu().numpy()

            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(img * 0.5 + 0.5); plt.axis("off")
            plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(gt, cmap="gray"); plt.axis("off")
            plt.subplot(1, 3, 3); plt.title("Prediction w/ Errors")
            plt.imshow(pred, cmap="gray")
            plt.imshow(fp_img, cmap="Reds", alpha=0.3)
            plt.imshow(fn_img, cmap="Blues", alpha=0.3)
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"inference_{i}.png"))
            plt.close()
            if i == 2:
                break

    with open(os.path.join(args.output_dir, "result.csv"), "w") as f:
        f.write(f"batch_size,lr,epochs,best_dice\n")
        f.write(f"{args.batch_size},{args.lr},{args.epochs},{best_metric:.4f}\n")

if __name__ == "__main__":
    main()