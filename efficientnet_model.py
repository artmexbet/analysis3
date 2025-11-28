"""
EfficientNet-based model for food vs non-food classification using transfer learning.
"""
import json
import os
import dvc.api
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_loader import load_datasets

# Load parameters from DVC
params = dvc.api.params_show()

SEED = params.get("seed", 42)
IMG_SIZE = params.get("img_size", 128)
BATCH_SIZE = params.get("batch_size", 32)
EPOCHS = params.get("efficientnet", {}).get("epochs", 5)

# Set random seed for reproducibility
tf.random.set_seed(SEED)

# Load data
train_ds, val_ds, test_ds, class_names = load_datasets(
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

print(f"Class names: {class_names}")

# Load pre-trained EfficientNetB0 model without top layers
base_model = keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze base model layers
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# EfficientNet has built-in preprocessing
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)

model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# Evaluate on test set
loss, accuracy, auc = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Save metrics
metrics = {
    "loss": float(loss),
    "accuracy": float(accuracy),
    "auc": float(auc),
}
with open("efficientnet.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/efficientnet.keras")
print("Model saved to models/efficientnet.keras")
