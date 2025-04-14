import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# ------------------------------
# 1. Dataset Loader
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
# 2. Autoencoder Model
# ------------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 256x256
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ------------------------------
# 3. Training with TensorBoard
# ------------------------------
def train_autoencoder():
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-3

    # Dataset
    transform = transforms.ToTensor()
    dataset = NoisyCleanPatchDataset("../dataset/patches/noisy", "../dataset/patches/ground_truth", transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device, model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(log_dir="./runs/autoencoder")

    # Log model graph
    sample_noisy, _ = next(iter(dataloader))
    writer.add_graph(model, sample_noisy.to(device))

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

    writer.close()
    torch.save(model.state_dict(), "autoencoder_denoiser.pth")
    print("Model saved and TensorBoard log complete.")

# ------------------------------
# 4. Run Training
# ------------------------------
if __name__ == "__main__":
    train_autoencoder()