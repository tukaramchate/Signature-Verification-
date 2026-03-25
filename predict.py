import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from inference_utils import load_best_threshold
from preprocess import preprocess_signature

THRESHOLD = load_best_threshold(default=0.75)
model = load_model("best_model.h5")


def predict_signature(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = preprocess_signature(img)
    img = np.expand_dims(img, axis=0)  # (1, 128, 128, 1)

    score = model.predict(img, verbose=0)[0][0]

    if score >= THRESHOLD:
        print(f"GENUINE  (confidence: {score * 100:.1f}%, threshold: {THRESHOLD:.2f})")
    else:
        print(f"FORGED   (confidence: {(1 - score) * 100:.1f}%, threshold: {THRESHOLD:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict whether a signature is genuine or forged.")
    parser.add_argument("image_path", help="Path to a signature image file (png/jpg/jpeg).")
    args = parser.parse_args()

    predict_signature(args.image_path)
