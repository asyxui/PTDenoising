import os
from PIL import Image

# Input folders
noisy_dir = "./noisy"
gt_dir = "./ground_truth"

# Output folders
patch_noisy_dir = "./patches/noisy"
patch_gt_dir = "./patches/ground_truth"
os.makedirs(patch_noisy_dir, exist_ok=True)
os.makedirs(patch_gt_dir, exist_ok=True)

# Patch size
patch_size = 256
skip_counter = 0

def create_patches(input_dir, output_dir, tag):
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path)

        if img.size != (1024, 1024):
            print(f"Skipping {img_file} — not 1024x1024.")
            skip_counter += 1
            continue

        base = os.path.splitext(img_file)[0]

        patch_id = 0
        for row in range(0, 1024, patch_size):
            for col in range(0, 1024, patch_size):
                patch = img.crop((col, row, col + patch_size, row + patch_size))
                patch_filename = f"{base}_{tag}_patch_{patch_id}.png"
                patch.save(os.path.join(output_dir, patch_filename))
                patch_id += 1

        print(f"Processed {img_file} -> {patch_id} patches")

# Create patches independently
create_patches(noisy_dir, patch_noisy_dir, "noisy")
create_patches(gt_dir, patch_gt_dir, "clean")

print(f"All patches generated. {skip_counter} images were skipped.")
