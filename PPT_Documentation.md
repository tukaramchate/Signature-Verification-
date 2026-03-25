# Signature Verification using CNN

## Slide 1: Title
- Project: Signature Verification using Convolutional Neural Network (CNN)
- Domain: Deep Learning / Computer Vision
- Goal: Classify uploaded signature as Genuine or Forged
- Tech stack: Python, TensorFlow/Keras, OpenCV, Streamlit

## Slide 2: Problem Statement
- Manual signature verification is slow and error-prone.
- Human inspection can fail on skilled forgeries.
- A reliable automated system is needed for fast verification.
- This project uses deep learning to detect forged signatures.

## Slide 3: Objectives
- Build an end-to-end signature verification pipeline.
- Train a CNN model on genuine and forged signatures.
- Provide real-time prediction via command line and web app.
- Improve decision quality using threshold calibration.

## Slide 4: Dataset
- Source: Signature dataset (CEDAR-style arrangement)
- Folder structure:
  - dataset/genuine -> real signatures
  - dataset/forged -> forged signatures
- Data used in workspace:
  - Genuine samples: 1139
  - Forged samples: 1010

## Slide 5: Project Architecture
- preprocess.py
  - Image preprocessing + train/test split
- model.py
  - CNN architecture definition
- train.py
  - Training, evaluation, best model save, threshold tuning
- predict.py
  - Single-image inference from terminal
- app.py
  - Streamlit web interface
- inference_utils.py
  - Loads tuned threshold from best_threshold.json

## Slide 6: Preprocessing Pipeline
- Convert image to grayscale
- Resize to 128 x 128
- Gaussian blur for mild denoising
- CLAHE for local contrast enhancement
- Otsu thresholding for binarization
- Normalize pixel values to [0, 1]
- Add channel dimension: (128, 128, 1)

## Slide 7: CNN Model Design
- Input layer: (128, 128, 1)
- Block 1: Conv2D(32) + BatchNorm + MaxPool
- Block 2: Conv2D(64) + BatchNorm + MaxPool
- Block 3: Conv2D(128) + BatchNorm + MaxPool
- Classifier: Flatten + Dense(256) + Dropout(0.5) + Dense(1, sigmoid)
- Output:
  - value near 1 -> Genuine
  - value near 0 -> Forged

## Slide 8: Training Strategy
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy, AUC
- Data augmentation:
  - rotation_range=10
  - width/height shift=0.1
  - zoom_range=0.1
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint (best_model.h5)

## Slide 9: Threshold Calibration
- Raw sigmoid output alone may not give best class balance.
- Threshold is tuned from 0.30 to 0.90 after training.
- Best threshold stored in best_threshold.json.
- Inference scripts load threshold automatically.
- Benefit: better trade-off between false genuine and false forged decisions.

## Slide 10: Evaluation Results (Current Workspace)
- Test Accuracy at threshold 0.50: 79.53%
- Calibrated threshold: 0.68
- Accuracy at calibrated threshold: 81.40%
- Balanced Accuracy at calibrated threshold: 81.50%

## Slide 11: Inference Flow
- User provides image via CLI or Streamlit app.
- Same preprocessing is applied as training.
- Model predicts genuine score.
- Score compared with tuned threshold.
- Final label + confidence is shown.

## Slide 12: Streamlit UI
- Upload signature image (png/jpg/jpeg)
- Click Verify Signature
- Displays:
  - Genuine/Forged label
  - Confidence
  - Raw genuine score
  - Progress indicator

## Slide 13: Tools and Libraries
- TensorFlow / Keras -> model training and inference
- OpenCV -> image preprocessing
- NumPy -> numerical operations
- scikit-learn -> split and utility operations
- Streamlit -> web UI
- Pillow -> image handling

## Slide 14: How to Run Project
- Install dependencies:
  - python -m pip install -r requirements.txt
- Train model:
  - python train.py
- Predict single image:
  - python predict.py "dataset/genuine/sample.png"
- Run web app:
  - python -m streamlit run app.py

## Slide 15: Limitations
- Accuracy depends on dataset quality and writer variability.
- Skilled forgeries can still be difficult.
- Current model is writer-independent, not identity-matching Siamese style.
- More robust generalization needs larger and more diverse data.

## Slide 16: Future Improvements
- Use Siamese/Triplet networks for pairwise signature matching.
- Add writer-wise split to avoid data leakage by signer similarity.
- Add advanced augmentations and regularization.
- Export training history graphs and confusion matrix.
- Package as API service for deployment.

## Slide 17: Demo Script (What to Say)
- "First, I train the CNN on genuine and forged signatures."
- "Then the best model and threshold are saved automatically."
- "Now I upload a signature in the Streamlit app."
- "The system preprocesses, predicts, and shows label with confidence."
- "This demonstrates a complete AI pipeline from data to deployment interface."

## Slide 18: Conclusion
- Built a complete signature verification system with CNN.
- Added threshold calibration to improve prediction balance.
- Integrated command-line and web-based inference.
- Project demonstrates practical AI engineering workflow end-to-end.
