# Skin Disease Detection using CNN | Streamlit Web App

## Project Overview

This project demonstrates a deep learning-based image classification model that can detect and classify 22 different types of skin diseases. The model is trained using TensorFlow/Keras and deployed via a Streamlit web application.

The goal is to make AI-driven dermatological assistance accessible to users without technical expertise, especially in regions with limited access to medical professionals. While discussing the idea with close friends and family, several expressed real interest in using such a tool, confirming the need and potential impact of the project.

## Objective

- Build a custom CNN model for skin disease detection.
- Deploy it using Streamlit for easy public access.
- Create an educational interface to showcase model performance and training.
- Serve as a foundational prototype for future full-stack deployment.

## Features

- Upload skin disease image and receive prediction.
- Model trained to identify 22 skin disease categories.
- Easy-to-use interface with direct deployment on Streamlit Cloud.
- Fully responsive and lightweight.

## Live Demo

[Visit the Live Web App]((https://skin-diseases-manasvinair.streamlit.app/))


## Dataset

- Consists of images categorized into 22 disease classes.
- Pre-split into `train` and `test` folders.
- Images are uniformly resized to 224x224 pixels.
- Loaded using `ImageDataGenerator` from Keras with augmentation.

## Model Details

- Framework: TensorFlow/Keras
- Input: 224x224 RGB images
- Output: 22-class softmax classification
- Architecture: Custom CNN / EfficientNet-based model
- Metrics: Accuracy, categorical cross-entropy

## Training Summary

- Data augmentation applied (rotation, zoom, shift)
- Optimizer: Adam
- Epochs: 25 (baseline)
- Training Accuracy: ~90%
- Validation Accuracy: ~85–90%
- Final Evaluation (test set): currently being optimized

## Streamlit App Features

- Upload `.jpg` or `.png` image via browser.
- Model returns prediction with confidence score.
- Minimal UI with fast loading and mobile responsiveness.
- No need to install anything locally—fully browser-based.

## Planned Enhancements

- Convert to full-stack web app using React (frontend), Node.js (backend), and MongoDB (database).
- Add user feedback collection and database logging.
- Visualize model training using accuracy/loss plots.
- Support multiple image upload and batch predictions.
- Integrate model explainability using Grad-CAM.
- Improve accuracy with advanced data preprocessing and fine-tuned models.

## Learning Goals

This project helps:
- Understand how CNNs are built and trained.
- Learn image preprocessing, augmentation, and classification techniques.
- Explore real-world deployment using Streamlit.
- Serve as a base for scaling into full production-ready applications.


