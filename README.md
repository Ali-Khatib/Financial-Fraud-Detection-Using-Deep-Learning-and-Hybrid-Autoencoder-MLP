# Financial Fraud Detection using Deep Learning and Hybrid Autoencoder–MLP

This project implements a machine learning–based **financial fraud detection system** designed to identify fraudulent transactions in highly imbalanced datasets. It combines traditional supervised deep learning with anomaly detection techniques to improve fraud detection performance. The system is built using Python, TensorFlow/Keras, and scikit-learn, and is intended for educational, experimental, and portfolio purposes.

The project explores two modeling approaches: a baseline Deep Neural Network (MLP) and an advanced Hybrid Autoencoder + MLP architecture. The hybrid approach leverages an autoencoder trained exclusively on non-fraudulent transactions to learn normal behavior patterns, which are then combined with supervised learning to improve fraud detection.

## Features

- End-to-end fraud detection pipeline
- Extensive exploratory data analysis (EDA) and visualization
- Custom feature engineering based on transaction behavior
- Handling of extreme class imbalance
- Two models implemented:
  - Deep Neural Network (MLP)
  - Hybrid Autoencoder + MLP
- Latent feature extraction and reconstruction error–based anomaly detection
- Evaluation using AUC, precision, recall, and confusion matrix
- Early stopping, regularization, and dropout for model stability

## System Workflow

Transaction Dataset  
→ Data Cleaning & Exploration  
→ Feature Engineering  
→ Train / Test Split  
→ Feature Scaling  
→ Model Training  
→ Prediction  
→ Evaluation & Visualization  

## Models Implemented

Baseline MLP  
A fully connected deep neural network trained using binary cross-entropy loss, class weighting, dropout, and L1 regularization to handle class imbalance and overfitting.

Hybrid Autoencoder + MLP  
An autoencoder is trained on non-fraud transactions to learn normal transaction representations. Latent features and reconstruction error are combined with original features and passed into a supervised MLP classifier for final fraud prediction.

## Feature Engineering

- Balance change features:
  - orig_balance_change
  - dest_balance_change
- Ratio-based features:
  - amount_ratio_orig
  - amount_ratio_dest
- One-hot encoding for transaction type
- Standard scaling for numeric features

## Evaluation Metrics

- AUC (primary metric)
- Precision
- Recall
- F1-score
- Confusion Matrix visualization

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- KaggleHub
- Jupyter Notebook

## Project Structure

Financial-Fraud-Detection/
├── notebook/
│   └── fraud_detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore

## How to Run

pip install -r requirements.txt  
jupyter notebook  

Open and run:
notebook/fraud_detection.ipynb

## Learning Outcomes

- Handling highly imbalanced datasets
- Applying deep learning to tabular financial data
- Using autoencoders for anomaly detection
- Combining unsupervised and supervised learning
- Evaluating fraud detection systems with proper metrics
- Building scalable ML pipelines

## Author

Ali Khatib  
AI / Machine Learning & Software Engineering  

## License

This project is intended for educational and experimental use.
