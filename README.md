# 🌿 Plant Disease Detection: Classical ML vs. Deep Learning

## **Project Title & Team Members**
**Title**: CIFAR-Based Plant Disease Classification: Classical Machine Learning vs. Deep Learning (CNN)  
**Team Members**:
- **Muhammad Mutaal Khan** (CMS ID: 522455) – Deep learning implementation, CNN architecture, overall pipeline.
- **Saif Ullah Farooqi** (CMS ID: 511676) – Classical ML (SVM, RF), evaluation, and visualization.

---

## **Abstract**
This project investigates the performance of **classical machine learning** (SVM, Random Forest) compared to a custom **Convolutional Neural Network (CNN)** architecture for the task of plant leaf disease identification. Using the **PlantVillage dataset** (54,304 images), we evaluated models on their ability to classify 38 distinct health categories. Results demonstrate that while SVM provides a strong classical baseline, the CNN model achieves a state-of-the-art accuracy of **98.88%**, proving the necessity of spatial feature learning for high-complexity agricultural vision tasks.

---

## **Introduction**
Early and accurate identification of plant diseases is a cornerstone of modern precision agriculture. This project aims to:
1.  Implement **Support Vector Machine (SVM)** and **Random Forest (RF)** using flattened pixel data.
2.  Develop a high-performance **CNN** using TensorFlow/Keras to capture spatial hierarchies.
3.  Establish a **fair comparative framework** to analyze the trade-off between model complexity and diagnostic accuracy.
4.  Analyze the **business impact** of deploying these models in real-world farming environments.

---

## **Dataset Description**
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets)
- **Size**: 54,304 RGB images.
- **Image Dimensions**: $128 \times 128 \times 3$ (RGB).
- **Classes**: 38 categories (e.g., Apple Scab, Tomato Early Blight, Healthy Corn, etc.).

[Image of a collage of various plant leaves showing different types of diseases like leaf spot, rust, and blight, labeled with their respective categories]

### **Splits & Preprocessing**
- **Training/Test Split**: 80/20 ratio.
- **Normalization**: Rescaled pixel intensities to the $[0, 1]$ range.
- **Reshaping**: Images were flattened for SVM/RF and kept
