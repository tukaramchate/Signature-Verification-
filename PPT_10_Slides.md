# Signature Verification using CNN - 10 Slide PPT Content

## Slide 1: Title Slide
- Title: Signature Verification using CNN
- Subtitle: Deep Learning-based Genuine vs Forged Signature Classification
- Name, Department, College
- Date

## Slide 2: Problem Statement
- Signature verification is still widely used in banking, legal, and academic workflows.
- Manual verification is slow and can be inconsistent.
- Skilled forgeries are difficult to detect reliably.
- Need an automated AI system for faster and more accurate verification.

## Slide 3: Project Objectives
- Build an end-to-end signature verification pipeline.
- Train a CNN model on genuine and forged signature images.
- Deploy prediction through CLI and web interface.
- Improve decision reliability with threshold calibration.

## Slide 4: Dataset and Structure
- Dataset classes:
  - genuine (real signatures)
  - forged (fake signatures)
- Workspace sample counts:
  - Genuine: 1139
  - Forged: 1010
- Folder structure:
  - dataset/genuine
  - dataset/forged

## Slide 5: Preprocessing Pipeline
- Convert to grayscale
- Resize to 128 x 128
- Gaussian blur for denoising
- CLAHE for contrast enhancement
- Otsu thresholding for binarization
- Normalize to [0, 1]
- Final tensor shape: (128, 128, 1)

## Slide 6: CNN Architecture
- Input: (128, 128, 1)
- Conv2D(32) + BatchNorm + MaxPool
- Conv2D(64) + BatchNorm + MaxPool
- Conv2D(128) + BatchNorm + MaxPool
- Flatten + Dense(256) + Dropout(0.5)
- Output layer: Dense(1, sigmoid)
- Output meaning:
  - score near 1 => genuine
  - score near 0 => forged

## Slide 7: Training Strategy
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Metrics: Accuracy, AUC
- Data augmentation: rotation, shift, zoom
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint (best_model.h5)

## Slide 8: Threshold Calibration and Inference
- Raw sigmoid threshold 0.50 is not always optimal.
- Post-training sweep from 0.30 to 0.90.
- Best threshold saved in best_threshold.json.
- Inference tools load threshold automatically:
  - predict.py
  - app.py
- Benefit: better balance between false accepts and false rejects.

## Slide 9: Results and Demo Flow
- Latest observed results in this workspace:
  - Test accuracy at threshold 0.50: 79.53%
  - Calibrated threshold: 0.68
  - Accuracy at 0.68: 81.40%
  - Balanced accuracy at 0.68: 81.50%
- Demo steps:
  - Train model
  - Predict single image
  - Upload image in Streamlit app and verify label/confidence

## Slide 10: Conclusion and Future Scope
- Built a complete AI pipeline: data -> preprocessing -> training -> inference -> UI.
- System successfully classifies signatures as genuine or forged.
- Threshold tuning improved practical decision quality.
- Future scope:
  - Siamese or triplet models for pairwise writer matching
  - More diverse datasets and writer-wise evaluation
  - API deployment for production use
