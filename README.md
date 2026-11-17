# Bangla Handwritten Word Recognition

**Name:** Faisal Ahmed
**Student ID:** 1000128819

Assignment 04 - Skills for Industry Competitiveness and Innovation Program (SICIP), BRAC University.

## Approach

The BanglaLekha-Isolated dataset contains 166,105 single handwritten characters across 84 classes (50 basic letters, 10 digits, and 24 compound characters), rather than whole words. To recognize complete handwritten Bangla words, this system implements a 5-stage pipeline:

1. **Character CNN (`train.py`)**: A convolutional neural network with data augmentation, 3 dual-Conv2D + BatchNormalization + MaxPooling + Dropout blocks, GlobalAveragePooling2D, and a Dense classifier trained on BanglaLekha-Isolated. Experiments are tracked in MLflow (hyperparameters, per-epoch metrics, final validation accuracy, and model artifacts).
2. **Word Segmentation (`segment.py`)**: The drawn word is binarized with Otsu thresholding, the top horizontal matra band is erased across sliding windows on a copy, low-hanging vowel strokes and narrow `া` (aa-kar) stems are filtered, and individual character intervals are cropped from the original drawing.
3. **Character Classification**: Each segmented character crop is centered and resized to match the training input resolution via `common.py` and passed through `models/model.keras`.
4. **Lexicon Word Matching (`lexicon.py`)**: Because the dataset lacks isolated vowel signs (kar), words in `words.txt` are normalized to their base-character sequences and matched against the CNN probability matrix using a probability-weighted dynamic programming edit distance.
5. **Interactive Web Interface (`app.py`)**: A Streamlit canvas application where users draw a Bangla word (such as মানুষ or জল) and view the top whole-word prediction alongside character-level debug crops.

## Project structure

```
bangla-ocr-assignment/
|-- train.py
|-- app.py
|-- common.py
|-- segment.py
|-- lexicon.py
|-- words.txt
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

## How to run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Train (at least two MLflow runs)
python train.py --data-dir data/Images --img-size 32 --lr 1e-3 --epochs 12 --run-name cnn-32px-lr1e-3
python train.py --data-dir data/Images --img-size 32 --lr 3e-4 --batch-size 64 --epochs 12 --run-name cnn-32px-lr3e-4-bs64

# 2) Start the MLflow UI
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000

# 3) Run the Streamlit application locally
streamlit run app.py

# 4) Build and run with Docker
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Then open `http://localhost:8501` in your browser.

## Results

| Run | Image size | Batch size | Learning rate | Epochs | Validation accuracy | Top-3 accuracy |
|-----|-----------|------------|---------------|--------|---------------------|----------------|
| `cnn-32px-lr1e-3` | 32x32 | 128 | 1e-3 | 12 | **92.97%** | 98.14% |
| `cnn-32px-lr3e-4-bs64` | 32x32 | 64 | 3e-4 | 12 | **92.62%** | 98.05% |

## Limitations

- Because BanglaLekha-Isolated does not include standalone vowel signs (kar), words are matched primarily on their base consonant sequences.
- Segmentation can degrade on heavily overlapping characters or words drawn without a clear matra.
- Mouse/touchpad drawings on a digital canvas have a slight domain gap compared to scanned pen-on-paper dataset samples.
