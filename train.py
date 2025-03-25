import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import os
import matplotlib.pyplot as plt

# MONAI imports
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.data import decollate_batch

# Import the desired model from model_[DesiredModel].py
from model_unetR import model
# Import the Dataset class from the transformation script
from transform_train_images import Dataset, PairedTransform 

# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Move model to GPU if available
model.to(device)

def parse_args():
    """Parse command-line arguments for easy hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Train a MONAI U-Net on a saved dataset.")

    parser.add_argument("--dataset_path", type=str, default="train_data_transformed.pth",
                        help="Path to the .pth file containing the saved dataset.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for the DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of worker processes for data loading.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Fraction of dataset for training (vs. validation).")
    parser.add_argument("--val_interval", type=int, default=2,
                        help="Validate every N epochs.")
    parser.add_argument("--output_dir", type=str, default="/home/alextu/scratch/DeepNucNet/results",
                        help="Directory to save outputs.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Define output directories (paths now set in one place)
    output_dir = os.path.join(args.output_dir, "saved_images")
    model_save_dir = os.path.join(args.output_dir, "best_metric_model_pths")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # Load the dataset from .pth
    print(f"[INFO] Loading dataset from: {args.dataset_path}")
    dataset = torch.load(args.dataset_path, weights_only=False)
    print(f"[INFO] Total samples: {len(dataset)}")

    # Split into train/validation
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Define loss, optimizer, and metrics
    loss_function = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Post-processing transforms for predictions & labels
    post_pred = Compose([EnsureType(), AsDiscrete(threshold=0.5)])
    post_label = Compose([EnsureType(), AsDiscrete()])

    # Training parameters
    max_epochs = args.epochs
    val_interval = args.val_interval

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []

    # Training Loop
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, (list)): #This is necessary only for UNET++ model, UNET++ tensor is contained in a list, we want the tensor only
                outputs = outputs[0]

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            val_steps = 0

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_inputs)

                    # Compute validation loss
                    loss = loss_function(val_outputs, val_labels)
                    val_loss += loss.item()
                    val_steps += 1

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                val_loss /= val_steps
                val_loss_values.append(val_loss)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_save_dir, "best_metric_model.pth"))

    # Save Training & Validation Loss and Validation Dice Score Plots
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.title("Epoch Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, color='blue', label='Training Loss')
    plt.plot([val_interval * (i + 1) for i in range(len(val_loss_values))], val_loss_values, color='red', label='Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Validation Mean Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.plot([val_interval * (i + 1) for i in range(len(metric_values))], metric_values, color='green', label='Validation Dice')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()

    # Perform inference and save images (UNCHANGED FROM YOUR ORIGINAL CODE)
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_image = val_data[0].to(device)
            val_label = val_data[1].to(device)
            val_output = model(val_image)
            val_output = torch.sigmoid(val_output)
            val_output = (val_output > 0.5).float()

            val_image_np = val_image[0].permute(1, 2, 0).cpu().numpy()
            val_label_np = val_label[0, 0].cpu().numpy()
            val_output_np = val_output[0, 0].cpu().numpy()

            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"Image {i}")
            plt.imshow(val_image_np * 0.5 + 0.5)
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title(f"Ground Truth {i}")
            plt.imshow(val_label_np, cmap="gray")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title(f"Model Output {i}")
            plt.imshow(val_output_np, cmap="gray")
            plt.axis("off")

            plt.savefig(os.path.join(output_dir, f"inference_{i}.png"))
            plt.close()

            if i == 2:
                break


if __name__ == "__main__":
    main()