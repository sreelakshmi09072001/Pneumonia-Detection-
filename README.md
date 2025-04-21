🫁 Pneumonia Detection using Convolutional Neural Networks (CNN)
📌 Overview
This project focuses on detecting Pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). Early and accurate detection of pneumonia can significantly improve treatment outcomes, and this project aims to leverage deep learning for that purpose.

📂 Dataset
The dataset used is the Chest X-Ray Images (Pneumonia) dataset from Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia. It contains:

Training Set: 5,216 images

Validation Set: 16 images

Test Set: 624 images

Each image is classified as either:

NORMAL (no pneumonia)

PNEUMONIA (infected lungs)

🧠 Model Architecture
The model is a custom CNN built from scratch using TensorFlow / Keras. The architecture includes:

Convolutional Layers (with ReLU activation)

Max Pooling Layers

Dropout for regularization

Fully Connected (Dense) Layers

Sigmoid Activation in Output Layer (binary classification)
