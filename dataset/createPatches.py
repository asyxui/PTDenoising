import os
from PIL import Image

# Base input and output folders
base_noisy_dir = "./noisy"
base_gt_dir = "./ground_truth"
patch_base_noisy_dir = "./patches/noisy"
patch_base_gt_dir = "./patches/ground_truth"

# Patch size
patch_size = 256
skip_counter = 0

# Create patches for each dataset split
splits = ["train", "val", "test"]

def create_patches(input_dir, output_dir):
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
                patch_filename = f"{base}_patch_{patch_id}.png"
                patch.save(os.path.join(output_dir, patch_filename))
                patch_id += 1

        print(f"Processed {img_file} -> {patch_id} patches")

# Create patches independently
for split in splits:
    noisy_dir = os.path.join(base_noisy_dir, split)
    gt_dir = os.path.join(base_gt_dir, split)
    patch_noisy_dir = os.path.join(patch_base_noisy_dir, split)
    patch_gt_dir = os.path.join(patch_base_gt_dir, split)

    os.makedirs(patch_noisy_dir, exist_ok=True)
    os.makedirs(patch_gt_dir, exist_ok=True)

    print(f"\nProcessing {split} set:")
    create_patches(noisy_dir, patch_noisy_dir)
    create_patches(gt_dir, patch_gt_dir)

print(f"All patches generated. {skip_counter} images were skipped.")
