import os
import json
import argparse
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ssim
from torch.utils.tensorboard import SummaryWriter
from optuna.integration.tensorboard import TensorBoardCallback

CONFIG_PATH = "./best_hyperparams.json"

# ------------------------------
# CLI Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
parser.add_argument("epochs", type=int, nargs="?", default=50, help="Number of epochs to train")
args = parser.parse_args()
USE_OPTUNA = args.tune
NUM_EPOCHS = args.epochs

# ------------------------------
# Dataset Loader
# ------------------------------
class NoisyCleanPatchDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform if transform else transforms.ToTensor()

        noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(".png")]
        clean_files = set(f for f in os.listdir(clean_dir) if f.endswith(".png"))

        self.data_pairs = []

        for noisy_file in noisy_files:
            parts = noisy_file.split("_")
            if len(parts) < 7:
                continue

            scene_name = parts[0]
            cam_idx = parts[2]
            patch_idx = parts[-1].replace(".png", "")
            clean_file = f"{scene_name}_cam_{cam_idx}_clean_patch_{patch_idx}.png"

            if clean_file in clean_files:
                noisy_path = os.path.join(noisy_dir, noisy_file)
                clean_path = os.path.join(clean_dir, clean_file)
                self.data_pairs.append((noisy_path, clean_path))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.data_pairs[idx]
        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        return self.transform(noisy_img), self.transform(clean_img)

# ------------------------------
# Autoencoder Model
# ------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DenoisingAutoencoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()

        self.encoder_blocks = nn.ModuleList([
            DoubleConv(3, base_channels),
            DoubleConv(base_channels, base_channels * 2),
            DoubleConv(base_channels * 2, base_channels * 4),
            DoubleConv(base_channels * 4, base_channels * 8),
        ])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
        ])
        self.decoder_blocks = nn.ModuleList([
            DoubleConv(base_channels * 16, base_channels * 8),
            DoubleConv(base_channels * 8, base_channels * 4),
            DoubleConv(base_channels * 4, base_channels * 2),
            DoubleConv(base_channels * 2, base_channels),
        ])

        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.activation = nn.Sigmoid()

        self.encoder = self._encode
        self.decoder = self._decode

    def _encode(self, x):
        self.skips = []
        for block in self.encoder_blocks:
            x = block(x)
            self.skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return x

    def _decode(self, x):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = self.skips[-(i + 1)]
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)
        return self.activation(self.final_conv(x))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ------------------------------
# Optuna Objective
# ------------------------------
def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    base_channels = trial.suggest_categorical("base_channels", [32, 64, 128])
    l1_weight = trial.suggest_float("l1_weight", 0.5, 1.0)
    beta1 = trial.suggest_float("beta1", 0.8, 0.99)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    eps = trial.suggest_float("eps", 1e-9, 1e-6, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    transform = transforms.ToTensor()
    dataset = NoisyCleanPatchDataset("../dataset/patches/noisy", "../dataset/patches/ground_truth", transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder(base_channels=base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=f"./runs/optuna_trial_{trial.number}")

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = l1_weight * nn.L1Loss()(outputs, clean_imgs) + (1 - l1_weight) * (1 - ssim(outputs, clean_imgs, data_range=1.0, size_average=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

    for key, val in trial.params.items():
        writer.add_scalar(f"Hyperparams/{key}", val, trial.number)
    writer.add_scalar("Trial/val_loss", avg_loss, trial.number)

    writer.close()
    return avg_loss

# ------------------------------
# Run Optuna & Final Training
# ------------------------------
if __name__ == "__main__":
    if USE_OPTUNA or not os.path.exists(CONFIG_PATH):
        study_name = "denoising_autoencoder_study"
        storage = "sqlite:///optuna_study.db"
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True)
        tb_callback = TensorBoardCallback("./optuna_tensorboard", metric_name="val_loss")
        study.optimize(objective, n_trials=30, callbacks=[tb_callback])

        best_params = study.best_trial.params
        with open(CONFIG_PATH, "w") as f:
            json.dump(best_params, f)
        print("Saved best hyperparameters to config file.")
    else:
        with open(CONFIG_PATH, "r") as f:
            best_params = json.load(f)
        print("Loaded best hyperparameters from config file.")

    transform = transforms.ToTensor()
    dataset = NoisyCleanPatchDataset("../dataset/patches/noisy", "../dataset/patches/ground_truth", transform)
    dataloader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder(base_channels=best_params["base_channels"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    writer = SummaryWriter(log_dir="./runs/final_training")

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = best_params["l1_weight"] * nn.L1Loss()(outputs, clean_imgs) + (1 - best_params["l1_weight"]) * (1 - ssim(outputs, clean_imgs, data_range=1.0, size_average=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

    writer.close()
