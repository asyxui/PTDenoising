import os
import argparse
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_msssim import ssim

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Test ONNX denoising model on test dataset.")
parser.add_argument("--model", type=str, default="./models/denoising_autoencoder.onnx", help="Path to the ONNX model file")
args = parser.parse_args()

# --- Directories ---
onnx_model_path = args.model
noisy_dir = "../dataset/patches/noisy/test"
clean_dir = "../dataset/patches/ground_truth/test"
comparison_dir = "onnx_outputs/comparisons"
denoised_dir = "onnx_outputs/denoised"
os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(denoised_dir, exist_ok=True)

# --- Image Transforms ---
transform = T.ToTensor()
to_image = T.ToPILImage()

# --- Load ONNX model ---
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Match noisy-clean pairs ---
noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(".png")]
clean_files = set(os.listdir(clean_dir))
data_pairs = []

for noisy_file in noisy_files:
    parts = noisy_file.split("_")
    if len(parts) < 7:
        continue
    scene_name = parts[0]
    cam_idx = parts[2]
    patch_idx = parts[-1].replace(".png", "")
    clean_file = f"{scene_name}_cam_{cam_idx}_clean_patch_{patch_idx}.png"
    if clean_file in clean_files:
        data_pairs.append((noisy_file, os.path.join(noisy_dir, noisy_file), os.path.join(clean_dir, clean_file)))

# --- Evaluation ---
results = []
for noisy_file, noisy_path, clean_path in tqdm(data_pairs, desc="Processing test set"):
    noisy_tensor = transform(Image.open(noisy_path).convert("RGB")).unsqueeze(0)  # [1, C, H, W]
    clean_tensor = transform(Image.open(clean_path).convert("RGB")).unsqueeze(0)

    # Metrics before denoising
    l1_noisy = torch.nn.functional.l1_loss(noisy_tensor, clean_tensor).item()
    ssim_noisy = ssim(noisy_tensor, clean_tensor, data_range=1.0).item()

    # ONNX inference
    denoised = session.run([output_name], {input_name: noisy_tensor.numpy()})[0]
    denoised_tensor = torch.from_numpy(denoised).clamp(0, 1)

    # Metrics after denoising
    l1_denoised = torch.nn.functional.l1_loss(denoised_tensor, clean_tensor).item()
    ssim_denoised = ssim(denoised_tensor, clean_tensor, data_range=1.0).item()

    # Record metrics
    results.append({
        "filename": noisy_file,
        "l1_noisy": l1_noisy,
        "l1_denoised": l1_denoised,
        "ssim_noisy": ssim_noisy,
        "ssim_denoised": ssim_denoised
    })

    # Save images
    noisy_img = to_image(noisy_tensor.squeeze(0))
    denoised_img = to_image(denoised_tensor.squeeze(0))
    clean_img = to_image(clean_tensor.squeeze(0))

    comparison_img = Image.new("RGB", (noisy_img.width * 3, noisy_img.height))
    comparison_img.paste(noisy_img, (0, 0))
    comparison_img.paste(denoised_img, (noisy_img.width, 0))
    comparison_img.paste(clean_img, (2 * noisy_img.width, 0))

    comparison_img.save(os.path.join(comparison_dir, noisy_file.replace(".png", "_compare.png")))
    denoised_img.save(os.path.join(denoised_dir, noisy_file.replace(".png", "_denoised.png")))

# --- Save CSV ---
df = pd.DataFrame(results)
df.to_csv("onnx_outputs/loss_metrics.csv", index=False)

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(df["l1_noisy"], label="L1 Noisy", color="red", alpha=0.6)
plt.plot(df["l1_denoised"], label="L1 Denoised", color="green", alpha=0.6)
plt.plot(df["ssim_noisy"], label="SSIM Noisy", color="orange", alpha=0.6)
plt.plot(df["ssim_denoised"], label="SSIM Denoised", color="blue", alpha=0.6)
plt.xlabel("Test Image Index")
plt.ylabel("Loss / Similarity")
plt.title("Comparison of L1 Loss and SSIM Before and After Denoising")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("onnx_outputs/loss_plot.png")
plt.close()

print("✓ CSV saved to onnx_outputs/loss_metrics.csv")
print("✓ Plot saved to onnx_outputs/loss_plot.png")
