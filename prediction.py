import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from config import settings

class BengaliWordPredictor:
    def __init__(self):
        self.model_path = settings.predict.model_path
        self.labels_json_path = settings.predict.labels_json_path
        self.image_size = settings.predict.image_size
        self.model = None
        self.index_to_label = {}

        self.load_model_and_labels()

    def load_model_and_labels(self):
        label_path = self.labels_json_path if os.path.exists(self.labels_json_path) else "labels.json"
        with open(label_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)

        if isinstance(label_mapping, list):
            self.index_to_label = {int(i): str(v) for i, v in enumerate(label_mapping)}
        else:
            self.index_to_label = {int(k): str(v) for k, v in label_mapping.items()}

        self.model = tf.keras.models.load_model(self.model_path)

        input_shape = self.model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape and len(input_shape) >= 3 and input_shape[1] is not None:
            self.image_size = int(input_shape[1])

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        # 1. Convert to grayscale numpy array
        gray = np.array(image.convert("L"), dtype=np.float32)

        # 2. INVERT: Make background Black (0) and Ink White (255)
        # The model was trained on black backgrounds; we must keep it this way.
        if gray.mean() > 127:
            gray = 255.0 - gray

        # 3. Tight-crop around the white ink
        ink_mask = gray > 40
        if np.any(ink_mask):
            rows = np.any(ink_mask, axis=1)
            cols = np.any(ink_mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            cropped = gray[y1:y2 + 1, x1:x2 + 1]
            ch, cw = cropped.shape
            
            # --- CRITICAL FIX: Proportional Square Padding ---
            target_side = int(max(ch, cw) * 1.30) 
            
            square = np.zeros((target_side, target_side), dtype=np.float32)
            y_off = (target_side - ch) // 2
            x_off = (target_side - cw) // 2
            
            # Paste the original, un-stretched crop into the center of the black square
            square[y_off:y_off + ch, x_off:x_off + cw] = cropped
            gray = square

        # 4. Resize to 32x32 using high-quality LANCZOS resampling
        pil_resized = Image.fromarray(gray.astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )

        # 5. Contrast Stretch: Force the surviving ink to be fully bright (255.0)
        arr_resized = np.array(pil_resized, dtype=np.float32)
        if arr_resized.max() > 0:
            arr_resized = (arr_resized / arr_resized.max()) * 255.0

        # 6. Expand dimensions for the CNN: (1, 32, 32, 1)
        # We pass the BLACK background (0) and WHITE ink (255.0) directly.
        # The model's internal Rescaling layer will divide it by 255.0 automatically.
        tensor = np.expand_dims(arr_resized, axis=0)
        tensor = np.expand_dims(tensor, axis=-1)
        
        return tensor

    def predict(self, image: Image.Image) -> tuple[str, float]:
        input_tensor = self.preprocess_image(image)

        predictions = self.model.predict(input_tensor, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        predicted_label = self.index_to_label.get(predicted_index, "?")

        return predicted_label, confidence