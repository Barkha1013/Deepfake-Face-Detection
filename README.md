# Deepfake Face Detection using Transfer Learning

## Project Overview

This project builds a deep learning system to detect **fake vs real face images** using transfer learning with ResNet18.

The model is trained on a dataset of real and manipulated facial images and can predict whether a face is authentic or AI-generated.

## Technologies Used

* Python
* PyTorch
* ResNet18
* Scikit-learn
* Google Colab

## Dataset

Real and Fake Face Dataset

Total Images: **2041**

Train Set: **1632 images**
Validation Set: **409 images**

## Model Architecture

Transfer learning was applied using a pretrained **ResNet18** model.

The final fully connected layer was modified for **binary classification**:

* Fake Face
* Real Face

## Training Configuration

Epochs: **10**
Loss Function: **CrossEntropyLoss**
Optimizer: **Adam**

## Results

Validation Accuracy: **64%**

Evaluation metrics used:

* Precision
* Recall
* F1-score
* Confusion Matrix

## Example Output

Model Prediction Example:

REAL FACE
or
FAKE FACE

## Future Improvements

* Train on larger deepfake datasets
* Improve accuracy with deeper CNN models
* Deploy as a web application
* Add explainability using Grad-CAM

## Author

Machine Learning Project using PyTorch
