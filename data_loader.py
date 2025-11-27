"""
Data loading utilities for butterfly classification.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


DATA_DIR = "./data"


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
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "valid")
    test_dir = os.path.join(DATA_DIR, "test")
    
    img_shape = (img_size, img_size)
    
    # Use categorical for multi-class classification
    label_mode = "categorical"
    
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
