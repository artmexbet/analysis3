"""
Simple CNN model for butterfly classification.
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
EPOCHS = params.get("cnn", {}).get("epochs", 5)

# Set random seed for reproducibility
tf.random.set_seed(SEED)

# Load data
train_ds, val_ds, test_ds, class_names = load_datasets(
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
)

num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Build simple CNN model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Normalize pixel values
x = layers.Rescaling(1.0 / 255)(inputs)

# Convolutional layers
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)

# Dense layers
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# Evaluate on test set
loss, accuracy = model.evaluate(test_ds)
print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")

# Save metrics
metrics = {"loss": float(loss), "accuracy": float(accuracy)}
with open("cnn.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/cnn.keras")
print("Model saved to models/cnn.keras")
