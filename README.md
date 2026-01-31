# Handwritten Character Recognition  
CodeAlpha – Machine Learning Internship (Task 3)

## Project Description
This project implements a **Handwritten Character Recognition** system using **image processing and deep learning techniques**. The system is designed to identify handwritten digits from images by learning visual patterns using a Convolutional Neural Network (CNN). This project was completed as part of the Machine Learning Internship at CodeAlpha.

## Objective
To build a deep learning–based model that accurately recognizes handwritten characters or digits from image data.

## Approach
The handwritten images are processed as grayscale images and normalized before being passed to a **Convolutional Neural Network (CNN)**. CNN layers automatically learn spatial features from images, making them highly effective for handwritten character recognition tasks.

## Tools and Technologies
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

## Dataset
This project uses the **MNIST dataset**, which contains handwritten digit images (0–9).  
The dataset is directly loaded using TensorFlow/Keras and does not require manual download.

Dataset source:
https://en.wikipedia.org/wiki/MNIST_database

## Methodology
The workflow of the project includes loading the MNIST dataset, normalizing image pixel values, reshaping data for CNN input, training a convolutional neural network, and evaluating the model on test data. The trained model predicts the digit present in a handwritten image.

## Model Evaluation
The model performance is evaluated using accuracy. The trained CNN achieves high accuracy (around 98–99%) on the MNIST test dataset, indicating effective handwritten digit recognition.

## Results
The model successfully recognizes handwritten digits and displays sample predictions along with the predicted digit label. The high accuracy demonstrates the effectiveness of CNNs for handwritten character recognition.

## How to Run
Install required libraries:
Run the Python script or Jupyter Notebook included in this repository.

## Summary
A CNN-based handwritten character recognition system was developed using the MNIST dataset. The model effectively classifies handwritten digits with high accuracy, demonstrating the power of deep learning in image-based recognition tasks.

## Author
Chandru  
Machine Learning Intern – CodeAlpha

