import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor

PATCH_SIZE = 256

# Load ONNX model
ort_sess = ort.InferenceSession("./models/denoising_autoencoder.onnx")

# Store zoom level and pan offset
zoom_factor = 1.0
zoom_step = 0.1
pan_offset = [0, 0]

def pad_image(img):
    h, w, _ = img.shape
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT), pad_h, pad_w

def denoise(img):
    padded_img, pad_h, pad_w = pad_image(img)
    h, w, _ = padded_img.shape
    output = np.zeros_like(padded_img, dtype=np.float32)

    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = padded_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE].astype(np.float32) / 255.0
            patch = patch.transpose(2, 0, 1)[np.newaxis, :]
            patch_out = ort_sess.run(None, {"input": patch})[0][0]
            patch_out = np.clip(patch_out.transpose(1, 2, 0), 0, 1)
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

    screen_res = 1280, 720
    orig_disp, _ = resize_to_screen(original, screen_res)
    den_disp, _ = resize_to_screen(denoised, screen_res)

    window_name = "Original | Press any key to switch to denoised | Scroll to zoom"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal zoom_factor
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                zoom_factor = min(zoom_factor + zoom_step, 5.0)
            else:
                zoom_factor = max(zoom_factor - zoom_step, 1.0)

    cv2.setMouseCallback(window_name, on_mouse)

    show_denoised = False
    while True:
        img = den_disp if show_denoised else orig_disp
        zoomed = zoom_and_pan(img, zoom_factor, pan_offset)
        cv2.imshow(window_name, zoomed)
        key = cv2.waitKey(30)

        if key == 27: # ESC
            break
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

if __name__ == "__main__":
    select_and_process()
