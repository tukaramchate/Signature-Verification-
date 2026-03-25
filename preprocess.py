import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128


def preprocess_signature(gray_img):
    resized = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))

    # Mild denoising and local contrast enhancement improve pen-stroke visibility.
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    normalized = binary.astype(np.float32) / 255.0

    return np.expand_dims(normalized, axis=-1)  # (128, 128, 1)


def load_data(dataset_path):
    X, y = [], []

    for label, folder in enumerate(["forged", "genuine"]):
        path = os.path.join(dataset_path, folder)
        if not os.path.isdir(path):
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = preprocess_signature(img)

            X.append(img)
            y.append(label)  # 0=forged, 1=genuine

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("No valid images found. Check dataset/genuine and dataset/forged folders.")

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
