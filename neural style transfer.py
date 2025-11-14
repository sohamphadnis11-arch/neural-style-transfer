# -*- coding: utf-8 -*-
"""
Neural Style Transfer (Refactored)
Using TF-Hub Arbitrary Image Stylization
"""

# ============================================
# 📦 1) Install Dependencies
# ============================================
!pip install -q tensorflow tensorflow-hub pillow

# ============================================
# 📚 2) Imports
# ============================================
import io, os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files

print("TensorFlow:", tf.__version__)
print("TF-Hub:", hub.__version__)

# ============================================
# 🧰 3) Helper Functions
# ============================================

def load_local_image(path, max_dim=512):
    """Load and preprocess image from a local path."""
    img = Image.open(path).convert('RGB')
    img.thumbnail((max_dim, max_dim))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr[None, ...]

def load_remote_image(url, max_dim=512):
    """Load and preprocess image from an online URL."""
    import requests
    data = requests.get(url).content
    img = Image.open(io.BytesIO(data)).convert('RGB')
    img.thumbnail((max_dim, max_dim))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr[None, ...]

def display(content, style, result):
    """Show the images side by side."""
    plt.figure(figsize=(15, 5))
    titles = ["Content Image", "Style Image", "Stylized Output"]
    images = [content[0], style[0], result[0]]

    for i, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i + 1)
        plt.title(title)
        plt.imshow(np.clip(image, 0.0, 1.0))
        plt.axis("off")

    plt.show()

def save_output(img_tensor, filename="/content/stylized.jpg"):
    """Save the output stylized image."""
    img = np.clip(img_tensor[0] * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(filename)
    return filename

# ============================================
# 🧠 4) Load Stylization Model
# ============================================
MODEL_URL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
model = hub.load(MODEL_URL)
print("\nModel Loaded Successfully ✔")

# ============================================
# 📤 5) Image Selection Logic
# ============================================
def get_images(use_upload=True):
    """Choose between file upload or URL-based loading."""
    if use_upload:
        print("\nUpload content & style images (2 files required)")
        uploaded = files.upload()

        files_list = list(uploaded.keys())
        if len(files_list) < 2:
            raise Exception("Please upload at least 2 images (content THEN style).")

        content = load_local_image(files_list[0])
        style   = load_local_image(files_list[1])
        return content, style

    else:
        content_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Golden_Gate_Bridge%2C_SF_%28cropped%29.jpg/800px-Golden_Gate_Bridge%2C_SF_%28cropped%29.jpg"
        style_url = "https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Starry_Night.jpg"

        content = load_remote_image(content_url)
        style = load_remote_image(style_url)
        return content, style

# ============================================
# ▶️ 6) Main Execution
# ============================================
def run_style_transfer():
    # Choose method for loading images
    use_upload = True  # set False for URL images

    content_img, style_img = get_images(use_upload)

    # Stylize
    stylized = model(tf.constant(content_img), tf.constant(style_img))[0]

    # Display results
    display(content_img, style_img, stylized)

    # Save and download result
    output_path = save_output(stylized)
    print(f"\nSaved to: {output_path}")
    files.download(output_path)

# ============================================
# 🚀 Run program
# ============================================
run_style_transfer()
