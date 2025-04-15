import torch
from torchvision import transforms
from PIL import Image

from autoencoder import DenoisingAutoencoder

# ------------------------------
# 1. Load the model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load("./models/autoencoder_denoiser_L1SSIM_150.pth", map_location=device))
model.eval()

# ------------------------------
# 2. Load and preprocess a noisy image
# ------------------------------
image_path = "./data/noisy_corner.png"

transform = transforms.Compose([
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

# ------------------------------
# 3. Denoise the image
# ------------------------------
with torch.no_grad():
    output_tensor = model(input_tensor)

# ------------------------------
# 4. Convert and save/display
# ------------------------------
output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())

# Option A: Save it
output_path = "./data/L1SSIMLoss_Shallow/denoised_corner_150.png"
output_image.save(output_path)
print(f"Denoised image saved to {output_path}")