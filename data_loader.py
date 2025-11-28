"""
Data loading utilities for food vs non-food classification.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


DATA_DIR = "./data"


def _resolve_split_dir(*candidates):
    """Return the first existing directory that matches a split name."""
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"None of the split directories exist: {', '.join(candidates)}"
    )


def load_datasets(img_size=128, batch_size=32, seed=42):
    """
    Load train, validation and test datasets from data directory.
    
    Args:
        img_size: Size of images (height, width)
        batch_size: Batch size for training
        seed: Random seed for reproducibility
    
    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    train_dir = _resolve_split_dir("train", "training")
    val_dir = _resolve_split_dir("valid", "validation")
    test_dir = _resolve_split_dir("test", "evaluation")

    img_shape = (img_size, img_size)

    # Use binary labels for the food vs non-food task
    label_mode = "binary"

    train_ds = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=img_shape,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    
    val_ds = image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=img_shape,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    
    test_ds = image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode=label_mode,
        image_size=img_shape,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    
    class_names = train_ds.class_names
    
    return train_ds, val_ds, test_ds, class_names
