This repository contains a deep learning project for classifying dog emotions from images. The dataset includes four emotion classes: angry, happy, relaxed, and sad, with a total of 4000 images. The project was developed for the CENG 476 – Introduction to Deep Learning course.

The main objective of this project is not only to obtain high accuracy, but also to understand model behavior, analyze overfitting and underfitting, and improve performance through systematic experimentation.

Project Overview

The project initially started with a custom Convolutional Neural Network (CNN) designed from scratch. Although the baseline model was able to learn basic visual patterns, its performance was limited. Validation accuracy remained around 40 percent, and the model showed strong confusion between certain classes, especially misclassifying angry dogs as sad.

During training, training accuracy continued to increase while validation accuracy stagnated and validation loss slightly increased. This behavior clearly indicated overfitting and weak generalization.

To overcome these limitations, the project was extended using transfer learning with a pretrained ResNet backbone. This change significantly improved feature extraction and overall performance. Several experiments were then conducted to better control overfitting and understand how different training strategies affect generalization.

Model Development Process

The following steps summarize the main development process:

First, a baseline CNN model was trained. This model suffered from underfitting and class confusion, especially for the angry class.

Next, the architecture was changed to a transfer learning approach using a pretrained ResNet model. This resulted in a large improvement in validation accuracy, but introduced strong overfitting.

To reduce overfitting, different strategies were applied, including training adjustments and careful monitoring of validation loss. This produced the best-performing model, achieving approximately 86 to 87 percent validation accuracy.

Additional experiments were conducted using different freezing strategies. Full freezing limited the model’s ability to adapt to the dataset and resulted in lower performance. Partial freezing provided a better balance but did not outperform the best non-frozen configuration.

Heavy data augmentation was also tested. While augmentation improved robustness, overly aggressive augmentation slightly reduced validation accuracy.

Final Results

The best model configuration was obtained using transfer learning combined with effective overfitting control strategies. This model achieved approximately 86 to 87 percent validation accuracy and showed stable validation loss behavior.

Training curves, confusion matrices, ROC curves, and detailed classification metrics are included in the repository to support the reported results and analysis.

How to Run the Code

This section explains how to run the project from scratch. No prior knowledge of the codebase is required.

Environment Setup

The project was developed using Python and PyTorch. Python version 3.9 or newer is recommended. Any terminal or command prompt can be used. An IDE is not required.

It is strongly recommended to create a virtual environment before installing dependencies to avoid conflicts with other Python projects.

Installing Dependencies

All required libraries are listed in the requirements.txt file. These include PyTorch, torchvision, numpy, pandas, scikit-learn, matplotlib, Pillow, tqdm, and Flask.

After activating the virtual environment, install all dependencies using the requirements file. Once installation is complete, the environment is ready to run the project.

Dataset Preparation

The dataset is not included in this repository and must be provided separately.

The training script expects the dataset to be organized in a single directory. Inside this directory, there must be a CSV file named labels.csv.

The labels.csv file must include at least two columns:
	•	a column containing image file names
	•	a column named “label” containing the emotion class

All image files referenced in labels.csv must be located in the same dataset directory.

The dataset directory path is defined inside the training script. If the dataset is stored in a different location, the dataset path in the training script should be updated accordingly.

Training the Model

To start training, run the training script.

This script performs the complete training pipeline, including:
	•	loading and preprocessing the dataset
	•	splitting data into training and validation sets
	•	training the model
	•	monitoring training and validation performance
	•	applying early stopping
	•	saving the best-performing model

During training, epoch-level information such as loss and accuracy for both training and validation sets is printed to the terminal.

At the end of training, the best model is saved automatically to the outputs directory.

Evaluating Results

After training is completed, the script generates several evaluation artifacts. These include training and validation accuracy curves, training and validation loss curves, a confusion matrix, ROC curves, and a classification report.

All evaluation files are saved in the project directory and can be used to analyze model performance and generalization behavior.

Running the Flask Application (Optional)

A Flask-based web application is included for inference.

Before running the Flask application, the training script must be executed at least once so that the trained model file exists.

After the model is saved, the Flask application can be started. Once the server starts, a local address is printed in the terminal. This address can be opened in a web browser.

From the web interface, a dog image can be uploaded and the predicted emotion class will be displayed.

Hardware Support and Reproducibility

The code automatically selects the best available hardware. If a CUDA-enabled GPU or Apple Silicon (MPS) device is available, it will be used. Otherwise, the code runs on CPU without requiring any changes.

Random seeds are fixed where possible to improve reproducibility, and all important training parameters are explicitly defined in the training script.

Authors

Yiğit Efe Ahi
Esmanur Tetik
