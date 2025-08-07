# PlantVillage
# 🌿 Plant Disease Classification using CNN | PlantVillage Dataset

This project uses Convolutional Neural Networks (CNNs) to classify images of plant leaves into healthy or diseased categories. The model is trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), which contains over 50,000 labeled images across various crop species and diseases.

---

## 🧠 Objectives

- Build a CNN-based image classifier for plant disease detection.
- Preprocess and augment image data to improve model generalization.
- Evaluate model performance using accuracy and confusion matrix.
- Identify class-wise performance issues and improve model robustness.

---

## 📁 Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Categories:** 38 classes including:
  - Apple___Black_rot
  - Corn___Common_rust
  - Grape___Esca_(Black_Measles)
  - Tomato___Healthy, etc.
- **Image Type:** RGB JPEG
- **Split:**
  - Training: 80%
  - Validation: 10%
  - Test: 10%

---

## 🔧 Tech Stack

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- Scikit-learn
- OpenCV (optional)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/plant_disease_cnn.git
   cd plant_disease_cnn

2. **Install dependencies:**
   pip install -r requirements.txt
3. **Add the dataset:**
   Download and extract the PlantVillage dataset.
   Organize it into train/, val/, and test/ directories.
4. **Run training notebook or script:**
   jupyter notebook plant_disease_classification.ipynb

## 📊 Model Performance

- Final Accuracy: ~XX% _(fill in after training)_
- Classes with highest and lowest accuracy
- Confusion Matrix and classification report

---

## ✅ Key Learnings

- Image classification for real-world agricultural problems
- Data augmentation and transfer learning
- Fine-tuning CNNs for multi-class problems
- Visualizing and interpreting model predictions

---

## 🧠 Future Improvements

- Add Grad-CAM visualizations for interpretability
- Deploy model using Streamlit or Flask
- Train using mobile-optimized models (e.g., MobileNet)
- Expand dataset with real-field images

---

## 🧑‍💻 Author

**Mansi Varshney**  
Connect on (https://www.linkedin.com/in/mansivarshney10)
