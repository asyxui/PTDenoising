import cv2
import piq
import numpy as np
import onnxruntime as ort
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

PATCH_SIZE = 256

# Load ONNX model to run on the GPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_sess = ort.InferenceSession("./models/denoising_autoencoder.onnx", providers=providers)

# Store zoom level and pan offset
zoom_factor = 1.0
zoom_step = 0.1
pan_offset = [0, 0]

def pad_image(img):
    h, w, _ = img.shape
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT), pad_h, pad_w

def infer_patch(patch):
    patch_in = patch.astype(np.float32) / 255.0
    patch_in = patch_in.transpose(2, 0, 1)[np.newaxis, :]
    patch_out = ort_sess.run(None, {"input": patch_in})[0][0]
    return np.clip(patch_out.transpose(1, 2, 0), 0, 1)

def denoise(img):
    padded_img, pad_h, pad_w = pad_image(img)
    h, w, _ = padded_img.shape
    output = np.zeros_like(padded_img, dtype=np.float32)

    patches = []
    coords = []

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = padded_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patches.append(patch)
            coords.append((y, x))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(infer_patch, patches))

    for (y, x), patch_out in zip(coords, results):
        output[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = patch_out

    denoised = np.clip(output[:img.shape[0], :img.shape[1]] * 255, 0, 255).astype(np.uint8)
    return denoised

def resize_to_screen(img, screen_size):
    h, w = img.shape[:2]
    scale = min(screen_size[0] / w, screen_size[1] / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size), scale

def zoom_and_pan(img, zoom, offset):
    h, w = img.shape[:2]
    center = np.array([w // 2, h // 2]) + offset
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    x1 = max(center[0] - new_w // 2, 0)
    y1 = max(center[1] - new_h // 2, 0)
    x2 = min(x1 + new_w, w)
    y2 = min(y1 + new_h, h)
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

def show_images(original, denoised):
    zoom_factor = 1.0
    pan_offset = [0, 0]
    dragging = False
    drag_start = (0, 0)

    h, w = original.shape[:2]
    window_name = "Denoiser Viewer (Scroll to Zoom | Drag to Pan | Key to Toggle | s to save denoised image | ESC to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    show_denoised = False

    def on_mouse(event, x, y, flags, param):
        nonlocal zoom_factor, pan_offset, dragging, drag_start

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            pan_offset[0] -= dx
            pan_offset[1] -= dy
            drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                zoom_factor = min(zoom_factor + zoom_step, 5.0)
            else:
                zoom_factor = max(zoom_factor - zoom_step, 1.0)

    cv2.setMouseCallback(window_name, on_mouse)

    original_score = compute_brisque_score(original)
    denoised_score = compute_brisque_score(denoised)

    while True:
        img = denoised if show_denoised else original
        img_h, img_w = img.shape[:2]

        # Apply zoom
        display_w = int(img_w * zoom_factor)
        display_h = int(img_h * zoom_factor)
        resized = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

        # Apply pan
        center_x = display_w // 2 + pan_offset[0]
        center_y = display_h // 2 + pan_offset[1]

        # Get window size
        win_w = max(640, cv2.getWindowImageRect(window_name)[2])
        win_h = max(480, cv2.getWindowImageRect(window_name)[3])

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        x1 = center_x - win_w // 2
        y1 = center_y - win_h // 2
        x2 = x1 + win_w
        y2 = y1 + win_h

        # Clamp view window
        x1 = max(0, min(x1, resized.shape[1] - win_w))
        y1 = max(0, min(y1, resized.shape[0] - win_h))
        x2 = x1 + win_w
        y2 = y1 + win_h

        # Crop the zoomed image
        cropped = resized[y1:y2, x1:x2]

        # If cropped smaller than canvas (e.g., edges), pad it
        ch, cw = cropped.shape[:2]
        canvas[:ch, :cw] = cropped

        label = f"BRISQUE: {denoised_score:.2f} (lower is better)" if show_denoised else f"BRISQUE: {original_score:.2f} (lower is better)"
        cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):
            save_root = tk.Tk()
            save_root.withdraw()
            file_path = filedialog.asksaveasfilename(
                title="Save denoised image",
                initialfile="denoised_image.png",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg *.jpeg")],
            )
            save_root.destroy()
            if file_path:
                cv2.imwrite(file_path, denoised)
                print(f"Denoised image saved to {file_path}")
            else:
                print("Switch to denoised image first (press any key), then press S to save.")
        elif key != -1:
            show_denoised = not show_denoised

    cv2.destroyAllWindows()


def select_and_process():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select image", filetypes=[("Images", "*.png *.jpg *.jpeg")])
    if path:
        img = cv2.imread(path)
        denoised = denoise(img)
        show_images(img, denoised)

def compute_brisque_score(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)  # [1, 3, H, W], float32 in [0,1]
    return piq.brisque(img_tensor, data_range=1.0).item()

if __name__ == "__main__":
    select_and_process()
