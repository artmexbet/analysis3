"""
Data augmentation script for food vs non-food classification.
Creates augmented copies of training images.
"""
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
SRC_DIR = "./data/training"
OUT_DIR = "./data/training"
IMG_SIZE = 128
AUG_PER_IMAGE = 2  # Increased from 1 to 2 for more augmentation
SEED = 42

# Augmentation pipeline
augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.1, seed=SEED + 1),
        layers.RandomZoom(0.1, seed=SEED + 2),
        layers.RandomBrightness(0.1, seed=SEED + 3),
    ],
    name="augmentation",
)


def load_image(img_path):
    """Load and preprocess image."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_image(arr, out_path):
    """Save numpy array as image."""
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    Image.fromarray(arr).save(out_path, format="JPEG", quality=90)


def augment_dataset(src_dir, out_dir, aug_per_image=1):
    """Create augmented copies of all images in source directory."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    
    for class_name in sorted(os.listdir(src_dir)):
        class_src = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_src):
            continue
            
        class_out = os.path.join(out_dir, class_name)
        os.makedirs(class_out, exist_ok=True)
        
        for fname in sorted(os.listdir(class_src)):
            # Skip already augmented files
            if fname.startswith("aug_"):
                continue
                
            src_path = os.path.join(class_src, fname)
            if not os.path.isfile(src_path):
                continue
                
            try:
                img_arr = load_image(src_path)
            except Exception as e:
                print(f"Skipping {src_path}: {e}")
                continue
            
            # Get base name without extension
            base_name = os.path.splitext(fname)[0]
            ext = os.path.splitext(fname)[1] or ".jpg"
            
            # Create augmented copies
            for i in range(aug_per_image):
                batch = np.expand_dims(img_arr, axis=0)
                aug_batch = augmentation(batch, training=True)
                aug_img = aug_batch[0].numpy()
                
                out_fname = f"aug_{i:02d}_{base_name}{ext}"
                out_path = os.path.join(class_out, out_fname)
                
                # Handle duplicates
                if os.path.exists(out_path):
                    unique_id = rng.integers(1_000_000)
                    out_fname = f"aug_{i:02d}_{base_name}_{unique_id}{ext}"
                    out_path = os.path.join(class_out, out_fname)
                
                save_image(aug_img, out_path)
        
        print(f"Processed class: {class_name}")


if __name__ == "__main__":
    augment_dataset(SRC_DIR, OUT_DIR, AUG_PER_IMAGE)
    print("Augmentation complete.")
