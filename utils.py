import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import joblib
import os

def plot_training(history, path_prefix):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path_prefix + "_training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()


def reshape_for_2d_td_lstm(X, time_per_chunk):
    samples, time_steps, freq_bins, channels = X.shape
    assert time_steps % time_per_chunk == 0, "Il numero di time steps deve essere divisibile per time_per_chunk"

    time_chunks = time_steps // time_per_chunk


    X_new = X.reshape((samples, time_chunks, time_per_chunk, freq_bins, channels))
    return X_new

def normalize_features(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=(0, 1))
    std = np.std(X_train, axis=(0, 1)) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std

def apply_normalization(X, mean, std):
    return (X - mean) / (std + 1e-8)

def save_normalization_params(mean, std, filename):
    joblib.dump({"mean": mean, "std": std}, filename)

def load_normalization_params(filename):
    params = joblib.load(filename)
    return params["mean"], params["std"]