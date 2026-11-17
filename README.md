# Bangla Handwritten Character and Word Recognition System

**Name:** Faisal Ahmed  
**Student ID:** 1000128819  

Assignment 04 — Skills for Industry Competitiveness and Innovation Program (SICIP), BRAC University.

---

## Overview

This project implements an end-to-end Bangla Handwritten Recognition system utilizing the **BanglaLekha-Isolated** dataset. The architecture is built to recognize 84 distinct classes, consisting of basic consonants, vowels, numerals, and complex compound characters (*Juktobornos*). It features robust image preprocessing, multi-character word segmentation, an interactive Streamlit user interface equipped with drawing and erasing tools, and complete experiment tracking via **MLflow**.

---

## Architecture & Approach

1. **Preprocessing (`common.py`)**: To eliminate discrepancies between training data and live user drawings, a unified preprocessing pipeline is enforced. Images are converted to grayscale, dynamically inverted so that ink is white against a black background, tightly bounded around the ink strokes, padded proportionally to preserve natural aspect ratios, and rescaled using high-quality Lanczos interpolation.
2. **Convolutional Neural Network (`train.py`)**: A deep CNN built using TensorFlow/Keras. It includes data augmentation layers (random rotation, translation, and zoom), three progressive convolutional blocks (Conv2D + BatchNormalization + MaxPooling + Dropout), Global Average Pooling, and a Softmax output layer matching the 84 dataset classes.
3. **MLflow Experiment Tracking**: Fulfilling the assignment requirements, the training script programmatically executes **two distinct training runs** (`cnn-32px` and `cnn-64px`) with varied hyperparameters. Every epoch's metrics, parameters, and resulting model artifacts are logged to a local SQLite-backed MLflow tracking server.
4. **Word Segmentation (`segmentation.py`)**: For multi-character input, images undergo Otsu binarization and top-matra filtering to slice connected words into precise individual character bounding boxes.
5. **Interactive Web Application (`app.py`)**: A Streamlit interface supporting single-character and multi-character word modes, complete with pen and eraser utilities.

---

## MLflow Training & Experiment Setup

The assignment explicitly requires **at least two MLflow runs** with different hyperparameters. The included `train.py` automates this requirement by iterating through two independent configurations in a single execution:

* **Run 1 (`cnn-32px`)**:
  * Image Resolution: `32x32` pixels
  * Learning Rate: `1e-3`
  * Architecture: 3-block CNN with data augmentation and early stopping callbacks.
* **Run 2 (`cnn-64px`)**:
  * Image Resolution: `64x64` pixels
  * Learning Rate: `3e-4`
  * Architecture: Scaled feature maps with identical regularization.

The training script automatically compares the validation accuracy (`val_accuracy`) of both runs, logs parameters and metrics to MLflow (`artifacts/mlflow/mlflow.db`), and saves the absolute highest-performing model weights directly to `models/model.keras`.

---

## Project Structure

```text
bangla-ocr-assignment/
|-- train.py
|-- app.py
|-- common.py
|-- segmentation.py
|-- prediction.py
|-- config.py
|-- requirements.txt
|-- Dockerfile
|-- README.md
|-- github_link.txt
|-- labels.json
|-- models/
|   `-- model.keras
|-- artifacts/
|   `-- mlflow/
`-- screenshots/
    |-- streamlit_app.png
    `-- mlflow_experiment.png
```

---

## How to Run Locally

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Training & MLflow Logging
Run the training script pointing to your extracted dataset directory:
```bash
python train.py --data-dir "/mnt/c/Users/88019/Downloads/BanglaLekha-Isolated/Images"
```
*(This will execute both the 32px and 64px runs, log them to MLflow, and update `models/model.keras` with the best weights).*

### 3. View MLflow Dashboard
To inspect your two training runs, parameters, metrics, and validation curves:
```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000
```
Open `http://localhost:5000` in your browser.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

### 5. Run via Docker
```bash
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```

---

## Limitations

* **Class Scope**: Limited strictly to the 84 target classes present in the *BanglaLekha-Isolated* training distribution.
* **Diacritics (*Kars*)**: Standalone vowel modifiers (such as া, ি, ু) are handled via base-consonant isolation rather than independent multi-label character extraction.
* **Canvas Domain Gap**: Mouse or touchpad input can exhibit minor stroke variations compared to scanned handwriting datasets.