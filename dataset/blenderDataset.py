import bpy
import os
import glob

def load_blend_file(filepath):
    """Load a Blender scene file and reset previous data."""
    bpy.ops.wm.read_factory_settings(use_empty=True)  # Reset Blender state
    bpy.ops.wm.open_mainfile(filepath=filepath)
    print(f"Loaded: {filepath}")

def setup_cycles():
    """Initial setup of Cycles settings without modifying samples repeatedly."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'  # Change to 'CPU' if needed
    scene.cycles.use_denoising = False  # Default to off
    scene.cycles.use_adaptive_sampling = False  # Ensure consistency

    # Force Blender to recognize changes
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

def process_scenes(scene_folder, output_dir, noisy_samples_list=[2, 5, 10, 50], clean_samples=50):
    """Load each scene once and render multiple images without reloading."""
    blend_files = glob.glob(os.path.join(scene_folder, "*.blend"))

    for blend_file in blend_files:
        scene_name = os.path.splitext(os.path.basename(blend_file))[0]
        print(f"\nProcessing scene: {scene_name}")

        # Load scene ONCE
        load_blend_file(blend_file)
        setup_cycles()  # Set general Cycles settings (without modifying samples)

        # Create output directories
        noisy_dir = os.path.join(output_dir, "noisy")
        clean_dir = os.path.join(output_dir, "ground_truth")
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

        # Render multiple noisy images (without reloading)
        for samples in noisy_samples_list:
            render_and_save(noisy_dir, f"{scene_name}_noisy_{samples}.png", samples, denoise=False)

        # Render Ground Truth (denoised)
        render_and_save(clean_dir, f"{scene_name}_clean.png", clean_samples, denoise=True)

        print(f"Finished rendering for {scene_name}\n")

# Set paths
scene_folder = os.getcwd() + "/scenes"
output_directory = os.getcwd()

# Run batch processing
process_scenes(scene_folder, output_directory)
