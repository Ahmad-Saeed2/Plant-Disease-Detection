# ML Project for Plant Disease Detection

This repository contains a machine learning project focused on detecting plant diseases from leaf images. The model uses multiple classification algorithms, including **Support Vector Machine (SVM)**, **Random Forest (RF)**, and **Convolutional Neural Network (CNN)** to classify various plant diseases.

## Project Overview

The project includes:
- **Data Preprocessing**: Image resizing, PCA (Principal Component Analysis) for feature reduction.
- **Model Training**: Training of SVM, Random Forest, and CNN models on the dataset.
- **Evaluation**: The performance of the models is evaluated using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and confusion matrices.

## Dataset

The dataset used in this project is the **PlantVillage Dataset**, which contains images of plant leaves categorized by disease. The dataset includes various plant species such as tomato, apple, and grape with different disease conditions.

## Installation

To run this project, ensure you have the following libraries installed:

```bash
pip install scikit-learn numpy matplotlib tensorflow seaborn
