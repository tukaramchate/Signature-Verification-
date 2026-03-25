import json

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_cnn
from preprocess import load_data

X_train, X_test, y_train, y_test = load_data("dataset/")

# Data augmentation
# Signature-specific augmentation is intentionally mild to avoid changing identity cues.
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
)

model = build_cnn()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "AUC"])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_model.h5", save_best_only=True),
]

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

eval_metrics = model.evaluate(X_test, y_test, return_dict=True)
print(f"\nTest Accuracy: {eval_metrics['accuracy'] * 100:.2f}%")

# Tune decision threshold on held-out split and persist it for inference scripts.
probs = model.predict(X_test, verbose=0).ravel()
best_threshold = 0.5
best_bal_acc = 0.0

for t in np.linspace(0.3, 0.9, 61):
    pred = (probs >= t).astype(int)
    tp = np.sum((pred == 1) & (y_test == 1))
    tn = np.sum((pred == 0) & (y_test == 0))
    fp = np.sum((pred == 1) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))

    recall_genuine = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_forged = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (recall_genuine + specificity_forged)

    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = float(t)

with open("best_threshold.json", "w", encoding="utf-8") as f:
    json.dump({"threshold": best_threshold, "balanced_accuracy": best_bal_acc}, f, indent=2)

print(f"Best threshold saved: {best_threshold:.3f} (balanced_acc={best_bal_acc * 100:.2f}%)")
