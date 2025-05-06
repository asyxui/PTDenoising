import bpy
import os
import glob

def load_blend_file(filepath):
    """Load a Blender scene file and reset previous data."""
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset Blender state
    bpy.ops.wm.open_mainfile(filepath=filepath)
    print(f"Loaded: {filepath}")

def enable_gpu_rendering():
    """Set device type to OPTIX"""
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX' # or 'CUDA'
    prefs.get_devices()

    print("Compute Device Type:", prefs.compute_device_type)
    for device in prefs.devices:
        device.use = True
        print(f"{device.name} | Type: {device.type} | Use: {device.use}")

    bpy.context.scene.cycles.device = 'GPU'

def setup_cycles():
    """Initial setup of Cycles settings without modifying samples repeatedly."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU' # Change to 'CPU' if needed
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()

def render_and_save(output_path, filename, samples, denoise=False):
    """Render and save the image without reloading the scene."""
    scene = bpy.context.scene
    scene.cycles.samples = samples  # Set samples directly here
    scene.cycles.use_denoising = denoise  # Enable denoising if needed
    scene.render.filepath = os.path.join(output_path, filename)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    print(f"Rendering: {filename} with {samples} samples")
    bpy.ops.render.render(write_still=True)

    # Confirm image saved
    if os.path.exists(scene.render.filepath):
        print(f"Saved: {scene.render.filepath}")
    else:
        print(f"ERROR: Image {filename} was NOT saved!")

def process_split(split_folder, split_name, output_dir, noisy_samples_list=[2, 5, 10, 25, 50, 100, 200, 500], clean_samples=100):
    """Load each scene once and render multiple images without reloading."""
    blend_files = glob.glob(os.path.join(split_folder, "*.blend"))

    for blend_file in blend_files:
        scene_name = os.path.splitext(os.path.basename(blend_file))[0]
        print(f"\nProcessing scene: {scene_name} ({split_name})")

        load_blend_file(blend_file)
        enable_gpu_rendering()
        setup_cycles()

        noisy_dir = os.path.join(output_dir, "noisy", split_name)
        clean_dir = os.path.join(output_dir, "ground_truth", split_name)
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

        camera_objects = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']

        for i, camera in enumerate(camera_objects):
            bpy.context.scene.camera = camera
            bpy.context.view_layer.update()

            for samples in noisy_samples_list:
                render_and_save(noisy_dir, f"{scene_name}_cam_{i}_noisy_{samples}.png", samples, denoise=False)
            render_and_save(clean_dir, f"{scene_name}_cam_{i}_clean.png", clean_samples, denoise=True)

        print(f"Finished rendering for {scene_name} ({split_name})\n")

# Main execution
base_scene_folder = os.path.join(os.getcwd(), "scenes")
output_directory = os.getcwd()

for split in ["train", "val", "test"]:
    split_folder = os.path.join(base_scene_folder, split)
    if os.path.exists(split_folder):
        process_split(split_folder, split, output_directory)
    else:
        print(f"Skipping missing split folder: {split_folder}")
