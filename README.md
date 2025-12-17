# 🌿 Plant Health AI: Multi-Model Disease Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)](https://www.tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
Crop diseases are a major threat to food security. This project develops an automated diagnosis system using the **PlantVillage dataset**. We conduct a rigorous comparison between **Deep Learning (CNN)** and **Traditional Machine Learning (SVM, Random Forest)** to identify the most robust architecture for detecting 38 different plant disease categories.



---

## 📊 Dataset Highlights
- **Scope**: 54,304 high-resolution RGB images.
- **Diversity**: 14 crop species (Tomato, Potato, Grape, Apple, etc.).
- **Categories**: 38 classes (Diseased vs. Healthy).
- **Preprocessing**: 
  - Images resized to $128 \times 128$.
  - Pixel values normalized to $[0, 1]$.
  - One-hot encoding for 38 target labels.

---

## 🧠 Model Architectures

### 1. Convolutional Neural Network (CNN) - *The Top Performer*
Designed to learn spatial hierarchies automatically.
- **Convolution Layers**: For automated feature extraction (edges, textures, spots).
- **Max Pooling**: To reduce spatial dimensions and focus on prominent features.
- **Dropout**: Implemented at $0.5$ rate to ensure the model generalizes to new fields.



### 2. Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF).
- **Approach**: High-dimensional hyperplane separation.

### 3. Random Forest (RF)
- **Configuration**: 200 Estimators (Decision Trees).
- **Approach**: Ensemble voting based on flattened pixel intensities.

---

## 🏆 Performance Comparison
The results demonstrate a clear advantage for spatial-aware models (CNN) over pixel-independent models (SVM/RF).

| Metric | Random Forest | SVM | **CNN (Ours)** |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 67.33% | 85.40% | **98.88%** |
| **Macro F1-Score** | 0.55 | 0.81 | **0.98** |
| **Weighted F1-Score** | 0.64 | 0.85 | **0.99** |



### **Key Discovery**
The CNN model achieved near-perfect accuracy because it analyzes the **spatial relationship** between pixels (detecting the shape of a lesion), whereas SVM/RF treat pixels as a flat list, losing critical "shape" information.

---

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install tensorflow opencv-python matplotlib scikit-learn pandas
