# Plant Disease Detection Using Machine Learning & Deep Learning

## Abstract

This project explores the effectiveness of both **classical machine learning models** and a **deep learning model** for plant disease classification using leaf images. We implement and evaluate **Support Vector Machine (SVM)** and **Random Forest (RF)** models after preprocessing the images with resizing and feature extraction techniques such as **PCA (Principal Component Analysis)**. Additionally, we train a **Convolutional Neural Network (CNN)** to learn rich image representations directly from data. The performance of each model is assessed using accuracy, precision, recall, F1‑score, and confusion matrices.

The results indicate that while classical models perform reasonably well, the **CNN achieves superior performance**, demonstrating the strength of deep feature learning for complex image classification tasks.

---

## Introduction

Image classification is a central problem in computer vision with applications in agriculture, healthcare, automation, and more. In this project, the goal is to detect plant diseases from leaf images and compare the effectiveness of classical machine learning models versus deep learning.

Key objectives of this project:
1. Apply data preprocessing and dimensionality reduction for classical models.
2. Train and tune classical models (SVM and Random Forest).
3. Build and evaluate a deep learning Convolutional Neural Network (CNN).
4. Compare performance across models using standard evaluation metrics.
5. Discuss model strengths, weaknesses, and potential improvements.

---

## Dataset Description

- **Dataset Used**: PlantVillage (or similar plant disease image dataset)
- **Image Type**: RGB leaf images
- **Classes**: Multiple plant diseases plus healthy leaf categories
- **Splits**:
  - **Training set**: Used to train the models
  - **Validation set**: Used for tuning and early stopping
  - **Test set**: Used to evaluate final model performance

---

## Data Preprocessing

Preprocessing steps include:
- **Image resizing** to smaller dimensions suitable for classical models.
- **Normalization** of pixel values.
- **PCA (Principal Component Analysis)** for dimensionality reduction before training SVM and RF.

These steps help improve computational efficiency and model performance.

---

## Models Implemented

### 1. Support Vector Machine (SVM)

- Hyperparameters tuned using **GridSearchCV**
- RBF kernel tested with different values of `C` and `gamma`
- Achieved competitive accuracy with a balanced performance across many classes.

### 2. Random Forest (RF)

- Ensemble of decision trees
- Handles non‑linear class boundaries and multi‑class problems effectively.

### 3. Convolutional Neural Network (CNN)

- Built using TensorFlow / Keras
- Learns hierarchical features directly from image pixels
- Provides the best performance among all models for this task.

---

## Evaluation & Results

Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1‑score**
- **Confusion Matrix**

For example, the SVM model achieved:
- **Accuracy:** 85.4%
- Macro and weighted averages show balanced performance across classes.

Confusion matrices and classification reports were visualized to analyze strengths and weaknesses per class.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
