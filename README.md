# Fashion Item Prediction

## 📌 Project Overview
This project aims to classify fashion products based on their attributes such as:
- **Category** (e.g., Accessories, Apparel, Footwear)
- **Sub-category** (e.g., Bags, Jewellery, Shoes)
- **Gender** (e.g., Men, Women, Unisex)
- **Base Color** (e.g., Black, Red, Blue)
- **Season** (e.g., Summer, Winter, Spring, Fall)

A deep learning model is trained using a **Fashion Product Images Dataset** from Kaggle. The model takes an image as input and predicts the above attributes.

---

## 📂 Dataset Details
- Source: [Fashion Product Images Dataset (Kaggle)](https://www.kaggle.com)
- Images: High-quality product images with metadata (category, color, season, etc.)
- Format: `.csv` file with image filenames and corresponding labels

---

## 🛠️ Tech Stack
- **Framework**: TensorFlow/Keras, OpenCV
- **Preprocessing**: NumPy, Pandas
- **Model Architecture**: CNN (Convolutional Neural Networks)
- **Training Environment**: Kaggle Notebooks (GPU)
- **Deployment**: Streamlit (optional)

---

## 🚀 Model Training Steps
1. **Preprocessing:**
   - Resize images
   - Normalize pixel values
   - Convert labels to categorical format
2. **Model Architecture:**
   - CNN with multiple convolutional layers
   - Batch normalization & dropout for regularization
3. **Training:**
   - Train using Kaggle GPU
   - Save model as `fashion_classifier.h5`
4. **Evaluation:**
   - Validate on unseen test images
   - Calculate accuracy & loss
5. **Deployment:**
   - Load trained model
   - Predict fashion attributes for new images

---

## 📥 How to Use
### 1️⃣ Download the Trained Model
Download the trained model from Google Drive:  
[**Fashion Classifier Model**](https://drive.google.com/file/d/1PcKEgk9AxyCs_FtfqDBI7uGry2R02IVS/view?usp=sharing)

In this project the data processed BATCH_SIZE = 10000

For more accuracy You Can Increase the Numbers upto 80000

### 2️⃣ Upload an Image & Get Predictions
Run the function below in a Jupyter Notebook:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model("fashion_classifier.h5")

# Function to predict fashion attributes
def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found!")
        return
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)
    predictions = model.predict(img)
    print("Predicted Attributes:", predictions)

# Test with a sample image
predict_image("sample.png")
```

---

## 🔧 Troubleshooting
**1. Model predicts wrong attributes?**
- Ensure image preprocessing (size, normalization) is correct
- Try a more advanced model (ResNet, EfficientNet)
- Check for dataset imbalance

**2. Kaggle notebook crashes due to RAM issues?**
- Use smaller batch sizes while training
- Train on a subset of the dataset first

**3. Want to save the model to Google Drive?**
Run the following code:
```python
from google.colab import files
files.download('fashion_classifier.h5')
```

---

## 📜 License
This project is for educational purposes only. Dataset belongs to respective owners.

---

## 📩 Contact
For queries, feel free to reach out!

