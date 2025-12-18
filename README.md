# Plant Disease Detection Using Machine Learning & Deep Learning

## Project Overview

This project aims to develop a robust and efficient system for detecting plant diseases from images of plant leaves. Early and accurate detection of plant diseases is crucial for preventing large-scale crop damage, enhancing agricultural productivity, and supporting sustainable farming practices. We explore and compare the performance of three machine learning models: **Support Vector Machine (SVM)**, **Random Forest**, and **Convolutional Neural Network (CNN)** to tackle this problem.

The models are trained and evaluated on the **PlantVillage dataset**, a widely-used benchmark in plant disease classification. By comparing classical machine learning models with deep learning models, we aim to identify the most effective approach for this task.

---

## Dataset

The **PlantVillage dataset** is used for this project, which contains labeled leaf images from a variety of plant species affected by diseases. The dataset is widely used in plant disease classification tasks and includes various plant diseases, making it ideal for evaluating machine learning models.

- **Total Images**: 54,304
- **Classes**: 38 unique plant disease categories, as well as healthy leaves.
- **Image Type**: RGB images of plant leaves, with a variety of disease types affecting different plant species.

---

## Tools and Libraries

The following Python libraries were used for data processing, model implementation, and performance visualization:

- **TensorFlow & Keras**: For building and training the Convolutional Neural Network (CNN).
- **Scikit-learn**: For implementing SVM, Random Forest, and data splitting, as well as evaluating performance metrics.
- **Matplotlib & Seaborn**: For data visualization and plotting model performance, including confusion matrices.
- **NumPy**: For numerical operations and array manipulation.
- **PIL (Pillow)**: For image loading, manipulation, and preprocessing.
- **`glob` and `os`**: For file system operations and handling image paths.

---

## Methodology

### 1. Data Loading and Exploration

- Images were loaded from the **PlantVillage dataset**, and paths were mapped to their respective disease labels.
- Initial exploration involved checking the total number of images per class and ensuring proper label extraction from the dataset.

### 2. Data Processing

- **Dataset Splitting**: The dataset was divided into Training, Validation, and Test sets with an approximate 80/10/10 split.
- **Image Resizing**: Images were resized to `64x64` pixels for classical machine learning models (SVM, RF) to reduce computational complexity.
- **Normalization**: Pixel values were normalized to the range `[0, 1]`.
- **Feature Extraction**: For SVM and Random Forest, **Principal Component Analysis (PCA)** was applied to reduce the dimensionality of the flattened image data, capturing essential features while mitigating the curse of dimensionality.

### 3. Model Implementation and Evaluation

#### a. Support Vector Machine (SVM)

- **Description**: SVM is a powerful supervised learning algorithm that finds an optimal hyperplane to separate the classes.
- **Hyperparameter Tuning**: We used **GridSearchCV** to optimize the hyperparameters `C`, `gamma`, and `kernel` for the Radial Basis Function (RBF) kernel.
- **Key Results**:
  - **Test Accuracy**: 85.4%
  - Achieved balanced performance across several plant categories, but struggled with smaller or underrepresented classes.

#### b. Random Forest (RF)

- **Description**: An ensemble learning method that constructs multiple decision trees and combines them to improve generalization and reduce overfitting.
- **Parameters**: Configured with `n_estimators=200`, `max_depth=None`, and `min_samples_split=5`.
- **Key Results**:
  - **Test Accuracy**: 67.33%
  - Struggled significantly with certain classes, particularly when using PCA-reduced features, indicating limitations in capturing complex relationships.

#### c. Convolutional Neural Network (CNN)

- **Description**: CNNs are deep learning models specifically designed for image data, capable of learning hierarchical features directly from raw pixels.
- **Architecture**: The CNN consists of multiple `Conv2D` layers followed by `MaxPooling2D` layers for feature extraction, and `Dense` layers for final classification.
- **Training**: The model was trained for 20 epochs using the **Adam optimizer** and **SparseCategoricalCrossentropy** loss function.
- **Key Results**:
  - **Test Accuracy**: 98.88%
  - The CNN showed exceptional performance, demonstrating the superior capability of deep learning models in capturing complex image features.

---

## Model Performance Comparison

Below is a comparison of the performance of all three models on the test set:

| Model            | Accuracy | Macro Avg F1 | Weighted Avg F1 |
|------------------|----------|--------------|-----------------|
| **SVM**          | 85.4%    | 0.81         | 0.85            |
| **Random Forest**| 67.33%   | 0.55         | 0.64            |
| **CNN**          | 98.88%   | 0.98         | 0.99            |

### Observations:
- The **CNN model** significantly outperformed both the SVM and Random Forest models, with a test accuracy of **98.88%**.
- The **SVM** achieved a strong accuracy of **85.4%**, making it a competitive classical model.
- **Random Forest** struggled the most, with an accuracy of **67.33%**, likely due to the challenges of handling high-dimensional image data and reliance on PCA for feature extraction.

---

## Conclusion

The **Convolutional Neural Network (CNN)** outperformed both classical machine learning models, achieving an impressive test accuracy of **98.88%**. The CNN’s superior performance underscores the power of deep learning in learning complex features directly from images, making it highly suitable for applications like plant disease detection. 

While **SVM** and **Random Forest** provide good results for certain tasks, the **CNN’s deep learning capabilities** make it the most effective approach for this problem. The robust performance of the CNN model suggests it is highly suitable for real-world agricultural applications, where **automated plant disease diagnosis** can greatly assist in optimizing crop health monitoring.

---

## Future Work

- **Improved Data Augmentation**: Implement techniques like **rotation**, **zooming**, and **flipping** to improve generalization and address class imbalance.
- **Deeper CNN Architectures**: Explore more sophisticated models like **ResNet**, **VGG**, and **EfficientNet** for further performance improvements.
- **Transfer Learning**: Fine-tune pre-trained models such as **VGG16** or **InceptionV3** on the plant disease dataset to boost performance, especially in classes with limited data.
- **Ensemble Methods**: Combine the predictions of multiple models (SVM, RF, CNN) to further improve the accuracy and robustness of the system.
- **Model Deployment**: Implement this system into a real-time application, such as a mobile app for farmers to quickly detect diseases in the field using their smartphone camera.

---

## Applications

This system can be used in various agricultural settings, including:
- **Precision Agriculture**: Early disease detection helps in applying targeted treatments, reducing pesticide usage, and optimizing crop yield.
- **Mobile Apps for Farmers**: A mobile-based application for farmers to upload images of their plants and receive instant disease diagnosis.
- **Farm Automation**: Integrating this model with drones and automated systems for continuous crop health monitoring in large farms.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Next Steps:
1. You can **clone this repository** and follow the instructions to **train models**, or you can **modify the dataset** to further improve performance.
2. We encourage the contribution of **better architectures**, **new features**, and **data augmentation techniques** to improve model performance.
