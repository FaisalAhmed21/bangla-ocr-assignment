# Bangla Handwritten Word Recognition

**Name:** Faisal Ahmed  
**Student ID:** 1000128819  

Assignment 04 - Skills for Industry Competitiveness and Innovation Program (SICIP), BRAC University.

## Approach

The BanglaLekha-Isolated dataset contains 166,105 single handwritten characters across 84 classes (50 basic letters, 10 digits, and 24 compound characters/Juktobornos). To build a robust recognition system, this project implements an end-to-end pipeline:

1. **Character CNN (`train.py`)**: A convolutional neural network trained on the 84 classes of the BanglaLekha-Isolated dataset. It features data augmentation layers (random rotation, translation, zoom), 3 dual-Conv2D blocks with BatchNormalization, MaxPooling, and Dropout, followed by GlobalAveragePooling and Dense classification. Experiments and hyperparameters are automatically tracked and logged to MLflow.
2. **Word Segmentation (`segmentation.py`)**: For multi-character word mode, the canvas image is binarized using Otsu's thresholding. The system isolates character bounding boxes, safely separates connected characters along the top *Matra* line, and extracts individual character crops.
3. **Character Classification (`prediction.py`)**: Segmented or single character images are preprocessed identically to training data (via `common.py`)—performing grayscale conversion, ink inversion, proportional square padding to prevent aspect-ratio distortion, and high-quality Lanczos scaling to 32x32 pixels—before being classified by the trained model.
4. **Interactive Web Interface (`app.py`)**: A clean Streamlit application featuring both single-character and multi-character word recognition modes, complete with a drawing pen and a thick eraser tool to easily clean up strokes.

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

## How to Run

```bash
# 1) Set up environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2) Train model (Executes the automated 32px and 64px MLflow runs)
python train.py --data-dir "/mnt/c/Users/88019/Downloads/BanglaLekha-Isolated/Images"

# 3) Start the MLflow UI to view experiments
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000

# 4) Run the Streamlit application locally
streamlit run app.py

# 5) Build and run with Docker
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Open `http://localhost:8501` in your browser to interact with the application.

## Results & MLflow Tracking

The training script automatically logs two separate experiments with varying hyperparameters (`cnn-32px` and `cnn-64px`) to MLflow, tracking validation loss, validation accuracy, and storing model artifacts. The script automatically compares the validation accuracies and saves the highest-performing model to `models/model.keras`.

## Limitations

- The model is trained on the *BanglaLekha-Isolated* dataset, which contains 84 core classes (basic letters, numerals, and 24 common Juktobornos). Characters outside these 84 classes cannot be recognized.
- Standalone vowel diacritics (*Kars* like া, ি, ু) are out of scope for this isolated model architecture.
- Drawing consistency on a mouse or touchpad digital canvas may introduce a slight domain gap compared to physical pen-on-paper dataset samples.