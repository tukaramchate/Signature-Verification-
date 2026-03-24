# Signature Verification using CNN

This project classifies handwritten signatures into two classes:
- `genuine` (real)
- `forged` (fake)

It includes model training, threshold calibration, command-line prediction, and a Streamlit web app.

## Project structure

- `dataset/genuine/` : real signatures
- `dataset/forged/` : fake signatures
- `preprocess.py` : image preprocessing and stratified train/test split
- `model.py` : CNN architecture
- `train.py` : training, evaluation, and threshold tuning
- `predict.py` : single-image inference from disk
- `app.py` : Streamlit UI
- `inference_utils.py` : loads saved threshold for inference
- `best_model.h5` : trained model artifact (generated after training)
- `best_threshold.json` : calibrated threshold artifact (generated after training)
- `requirements.txt` : dependencies

## Current pipeline

### 1) Preprocessing

Each signature image is:
1. Converted to grayscale
2. Resized to `128 x 128`
3. Denoised with Gaussian blur
4. Contrast-enhanced with CLAHE
5. Binarized with Otsu thresholding
6. Normalized to `[0, 1]`

### 2) CNN model

Input shape: `(128, 128, 1)`

Architecture:
1. Conv2D(32) + BatchNorm + MaxPool
2. Conv2D(64) + BatchNorm + MaxPool
3. Conv2D(128) + BatchNorm + MaxPool
4. Flatten + Dense(256) + Dropout(0.5) + Dense(1, sigmoid)

### 3) Training strategy

- Loss: `binary_crossentropy`
- Metrics: `accuracy`, `AUC`
- Augmentation: mild rotation/shift/zoom for signatures
- Callbacks:
	- `EarlyStopping`
	- `ReduceLROnPlateau`
	- `ModelCheckpoint`

### 4) Threshold calibration

After training, `train.py` sweeps thresholds from `0.30` to `0.90` and saves the best threshold (based on balanced accuracy) to `best_threshold.json`.

Both `predict.py` and `app.py` load this threshold automatically.

## Setup

### Option A: use project virtual environment (recommended)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Option B: without activation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

### 1) Train model

```powershell
.\.venv\Scripts\python.exe train.py
```

### 2) Predict one image from terminal

```powershell
.\.venv\Scripts\python.exe predict.py "dataset\genuine\some_image.png"
```

### 3) Launch Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Note: use `streamlit run app.py` (or `python -m streamlit run app.py`), not `python app.py`.

## Dataset

Use a signature dataset such as CEDAR (Kaggle), and place images in:
- `dataset/genuine`
- `dataset/forged`

## Latest observed metrics in this workspace

From the latest completed training run:
- Test accuracy: `79.53%` at default threshold `0.50`
- Calibrated threshold: `0.68`
- Accuracy at calibrated threshold: `81.40%`
- Balanced accuracy at calibrated threshold: `81.50%`

Performance varies by dataset quality, split, and preprocessing.
# Signature-Verification-
