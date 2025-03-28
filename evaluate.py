# ============================
# evaluate.py
# ============================
import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import compute_hausdorff_distance
from monai.transforms import Compose, AsDiscrete, EnsureType
from monai.data import decollate_batch
from monai.losses import DiceLoss
from sklearn.metrics import precision_score, recall_score

from transform_train_test_images_with_augmentations import Dataset, PairedTransform

from model_unetR import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
model.to(device)

post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
post_label = Compose([EnsureType(), AsDiscrete()])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)  # SAFER FOR EVALUATION
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--max_visuals", type=int, default=5)
    return parser.parse_args()

def evaluate(model, dataloader, output_dir, max_visuals=5):
    model.eval()
    dice_scores, precisions, recalls, hausdorffs = [], [], [], []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image, label = batch[0].to(device), batch[1].to(device)

            output = model(image)
            if isinstance(output, list):
                output = output[0]
            pred_prob = torch.sigmoid(output)
            pred_bin = pred_prob > 0.5

            pred_post = [post_pred(i) for i in decollate_batch(pred_prob)]
            label_post = [post_label(i) for i in decollate_batch(label)]

            # Compute metrics
            y_true = label_post[0].cpu().numpy().flatten().astype(int)
            y_pred = pred_post[0].cpu().numpy().flatten().astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            dice = DiceLoss(sigmoid=False)(pred_post[0].unsqueeze(0), label_post[0].unsqueeze(0)).item()
            hd = compute_hausdorff_distance(pred_post[0].unsqueeze(0), label_post[0].unsqueeze(0), include_background=False)[0].item()

            precisions.append(precision)
            recalls.append(recall)
            dice_scores.append(1 - dice)  # DiceLoss = 1 - Dice
            hausdorffs.append(hd)

            # Visualize
            if i < max_visuals:
                img_np = image[0].permute(1, 2, 0).cpu().numpy()
                gt = label[0, 0].cpu().numpy()
                pred = pred_bin[0, 0].cpu().numpy()
                fp = ((pred == 1) & (gt == 0)).astype(np.uint8)
                fn = ((pred == 0) & (gt == 1)).astype(np.uint8)

                plt.figure(figsize=(18, 6))
                plt.subplot(1, 3, 1); plt.title("Image"); plt.imshow(img_np * 0.5 + 0.5); plt.axis("off")
                plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(gt, cmap="gray"); plt.axis("off")
                plt.subplot(1, 3, 3); plt.title("Prediction w/ Errors")
                plt.imshow(pred, cmap="gray")
                plt.imshow(fp, cmap="Reds", alpha=0.3)
                plt.imshow(fn, cmap="Blues", alpha=0.3)
                plt.axis("off")
                plt.savefig(os.path.join(output_dir, f"test_result_{i}.png"))
                plt.close()

    # Report metrics
    print("\n=== Evaluation Results on Test Set ===")
    print(f"Average Dice:       {np.mean(dice_scores):.4f}")
    print(f"Average Precision:  {np.mean(precisions):.4f}")
    print(f"Average Recall:     {np.mean(recalls):.4f}")
    print(f"Average Hausdorff:  {np.mean(hausdorffs):.4f}")

    with open(os.path.join(output_dir, "test_metrics.csv"), "w") as f:
        f.write("dice,precision,recall,hausdorff\n")
        for d, p, r, h in zip(dice_scores, precisions, recalls, hausdorffs):
            f.write(f"{d:.4f},{p:.4f},{r:.4f},{h:.4f}\n")
        f.write(f"\nMEAN,{np.mean(dice_scores):.4f},{np.mean(precisions):.4f},{np.mean(recalls):.4f},{np.mean(hausdorffs):.4f}\n")

if __name__ == "__main__":
    args = parse_args()

    # Correct: Load the already transformed test Dataset
    test_dataset = torch.load(args.test_data_path, weights_only=False)

    # Lower num_workers to avoid warnings/slowness
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    evaluate(model, test_loader, args.output_dir, args.max_visuals)
