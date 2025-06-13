# Acoustic Scene Classification (DCASE 2013)
This repository contains the code for an acoustic scene classification project based on the DCASE 2013 dataset. The system uses different Deep Learning architectures to classify audio recordings into 10 predefined scenes.

**Branch Structure**
The project is divided into two main branches to separate the two experimental phases:

`python-project`: Contains the project version without data augmentation. Features are pre-calculated once for faster training.

`python-project-data-augmentation`: Contains the project version with dynamic data augmentation. It uses a data generator to apply Time Stretching, Pitch Shifting, and Noise Addition "on-the-fly" during training.

**Usage**
The main scripts to interact with the project are `main.py` and `eval.py`.

**1. Model Training**
Running the `main.py` script starts the training process for the models specified within it.

```python main.py```

**Output**: During training, the best models (`.h5`) and training history plots (`_training_history.png`) are automatically saved in the `/results` folder.

**2. Model Evaluation**
Running the `eval.py` script evaluates the pre-trained models on the test set.

```python eval.py```

**Output:** The script generates and saves the evaluation results in the `/results` folder. This includes:

**A confusion matrix** (e.g., `_confusion_matrix.png`).

A text file with **performance metrics** (e.g., `_report.txt` with accuracy, precision, recall, F1-score).
