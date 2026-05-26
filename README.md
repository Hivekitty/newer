# Knowledge-Based Pest Management System

A CNN-based image classification model that identifies crop diseases and pests across **38 categories**, trained on 56,000+ images using TensorFlow/Keras.

---

## Dataset

- **Source:** [Plant Disease Dataset – Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Train set:** 56,251 images across 38 classes
- **Validation set:** 3,503 images across 38 classes
- **Input size:** 128 × 128 RGB

---

## Model Architecture

```
Input (128, 128, 3)
→ Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
→ Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
→ Flatten
→ Dense(128, relu) → Dropout(0.5)
→ Dense(38, softmax)
```

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Epochs:** 20

---

## Training Results

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------|---------------|--------------|----------|
| 1     | 14.53%        | 45.50%       | 1.8310   |
| 5     | 63.51%        | 78.08%       | 0.7151   |
| 10    | 72.57%        | 82.39%       | 0.5509   |
| 15    | 76.44%        | 84.36%       | 0.4915   |
| 20    | **78.92%**    | **87.21%**   | **0.3998** |

Final validation accuracy: **87.21%**

---

## Data Augmentation

Training images were augmented with:
- Rotation (±30°)
- Width/height shifts (±20%)
- Shear and zoom (±20%)
- Horizontal flip

---

## Files

| File | Description |
|------|-------------|
| `trained_model.ipynb` | Training notebook |
| `pest_management_model.keras` | Saved model (recommended) |
| `pest_management_model.h5` | Saved model (legacy HDF5) |

---

## Usage

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("pest_management_model.keras")

img = image.load_img("your_image.jpg", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
print(f"Predicted class index: {predicted_class}")
```

---

## Requirements

```
tensorflow>=2.10
numpy
```

Install with:

```bash
pip install tensorflow numpy
```

---

## Tech Stack

Python · TensorFlow · Keras · CNN · ImageDataGenerator
