# Brain-Tumor-Detector
An AI-powered web application that detects and classifies brain tumors from MRI images using deep learning.

## Overview

This project uses a trained neural network model to analyze brain MRI scans and classify them into one of four categories:

Glioma Tumor
Meningioma Tumor
Pituitary Tumor
No Tumor

------------------------------

The application is built with PyTorch for the model and Gradio for an interactive user interface.

## Features
Upload MRI images and get instant predictions
Classifies 4 tumor types
Displays prediction confidence scores
Simple and clean web interface
Fast inference
Demo

Upload an MRI image and the model will:

Predict the tumor type
Show confidence levels
Display the processed image
Installation

## Clone the repository:

git clone https://github.com/xo-source/Brain-Tumor-Detector-Ai.git
cd brain-tumor-detector-ai

## Install dependencies:

pip install -r requirements.txt

## Run the app:

python app.py
Requirements
Python 3.8+
torch
gradio
Pillow
numpy

## Model Details
Framework: PyTorch
Task: Image Classification
Classes: 4
Input: Brain MRI images

Note: Accuracy depends on dataset quality and evaluation method.

## Project Structure

brain-tumor-detector-ai/
│── app.py
│── model.pth
│── requirements.txt
│── README.md

## Disclaimer

This project is for educational and research purposes only.
It is not intended for medical diagnosis or clinical use.

## Future Improvements
Improve model generalization
Add tumor localization (bounding boxes or heatmaps)
Deploy as a public web app
Enhance UI/UX
