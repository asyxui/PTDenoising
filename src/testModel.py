import os
import argparse
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Test ONNX denoising model on test dataset.")
parser.add_argument(
    "--model", type=str,
    default="./models/denoising_autoencoder_400.onnx",
    help="Path to the ONNX model file"
)
args = parser.parse_args()

# --- Directories ---
onnx_model_path = args.model
noisy_dir = "../dataset/patches/noisy/test"
clean_dir = "../dataset/patches/ground_truth/test"
comparison_dir = "onnx_outputs/comparisons2"
denoised_dir = "onnx_outputs/denoised2"
os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(denoised_dir, exist_ok=True)

# --- Image Transforms ---
transform = T.ToTensor()
to_image = T.ToPILImage()

# --- Load ONNX model ---
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Build matching noisy-clean pairs ---
noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(".png")]
clean_files = set(f for f in os.listdir(clean_dir) if f.endswith(".png"))
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
        noisy_path = os.path.join(noisy_dir, noisy_file)
        clean_path = os.path.join(clean_dir, clean_file)
        data_pairs.append((noisy_file, noisy_path, clean_path))

# --- Run inference and save outputs ---
for noisy_file, noisy_path, clean_path in tqdm(data_pairs, desc="Processing test set"):
    noisy_tensor = transform(Image.open(noisy_path).convert("RGB")).unsqueeze(0).numpy()  # [1, C, H, W]
    clean_tensor = transform(Image.open(clean_path).convert("RGB"))

    # ONNX inference
    denoised = session.run([output_name], {input_name: noisy_tensor})[0]
    denoised_tensor = torch.from_numpy(denoised.squeeze(0)).clamp(0, 1)

    # Convert to images
    noisy_pil = to_image(torch.from_numpy(noisy_tensor.squeeze(0)))
    denoised_pil = to_image(denoised_tensor)
    clean_pil = to_image(clean_tensor)

    # Save comparison
    comparison_img = Image.new('RGB', (noisy_pil.width * 3, noisy_pil.height))
    comparison_img.paste(noisy_pil, (0, 0))
    comparison_img.paste(denoised_pil, (noisy_pil.width, 0))
    comparison_img.paste(clean_pil, (2 * noisy_pil.width, 0))
    comparison_name = noisy_file.replace(".png", "_compare.png")
    comparison_img.save(os.path.join(comparison_dir, comparison_name))

    # Save denoised-only image
    denoised_name = noisy_file.replace(".png", "_denoised.png")
    denoised_pil.save(os.path.join(denoised_dir, denoised_name))

print(f"Saved comparison images to: {comparison_dir}")
print(f"Saved denoised images to: {denoised_dir}")
