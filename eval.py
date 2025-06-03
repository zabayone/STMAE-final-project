import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import config
from data.data_loader import DataLoader
from data.feature_extractor import FeatureExtractor

def normalize_features(X, mean, std):
    return (X - mean) / (std + 1e-8)

def reshape_for_2d_td_lstm(X, time_per_chunk=7):
    samples, time_steps, freq_bins, channels = X.shape
    assert time_steps % time_per_chunk == 0, "Il numero di time steps deve essere divisibile per time_per_chunk"
    time_chunks = time_steps // time_per_chunk
    return X.reshape((samples, time_chunks, time_per_chunk, freq_bins, channels))

def plot_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    plt.close()

def evaluate_model(model_type, model_path, label_encoder, X_test, y_test_enc, results_dir):
    print(f"\n>> Valutazione del modello: {model_type}")

    model = load_model(model_path)

    test_loss, test_accuracy = model.evaluate(X_test, y_test_enc, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_, digits=4)
    print(report)

    with open(os.path.join(results_dir, f"{model_type}_report.txt"), "w") as f:
        f.write(f"Accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    plot_confusion_matrix(y_test_enc, y_pred, label_encoder.classes_, os.path.join(results_dir, f"{model_type}_confusion_matrix.png"))

def main():
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    print("Caricamento dataset di test...")
    loader = DataLoader(config.EVAL_DATASET_PATH, mono=config.MONO)
    X_test_raw, y_test = loader.load()

    print("Caricamento LabelEncoder...")
    label_encoder = joblib.load("label_encoder.pkl")
    y_test_enc = label_encoder.transform(y_test)

    print("Estrazione feature...")
    extractor = FeatureExtractor(sample_rate=44100, bands=config.BANDS, mode=config.SIGNAL_PROCESSING_TYPE)
    X_test = np.array(extractor.extract(X_test_raw))

    print("Normalizzazione...")
    history_path = os.path.join(config.RESULTS_PATH, config.MODEL_TYPE + "_history.pkl")
    assert os.path.exists(history_path), f"File di history non trovato: {history_path}"
    history = joblib.load(history_path)

    # Ricostruisci media e std dai dati normalizzati durante il training
    mean = np.mean(X_test, axis=0)
    std = np.std(X_test, axis=0)
    X_test = normalize_features(X_test, mean, std)

    MODELS_TO_EVALUATE = [
        "conv1d",
        "conv1d_td",
        "conv1d_td_lstm",
        "conv2d_td_lstm",
        "conv2d"
    ]

    for model_type in MODELS_TO_EVALUATE:
        print(f"\n--- Modello: {model_type} ---")

        model_file = os.path.join(config.RESULTS_PATH, model_type + "_best.h5")
        if not os.path.exists(model_file):
            print(f"⚠️  Modello non trovato: {model_file}")
            continue

        X_test_proc = X_test

        if model_type in ["conv1d_td", "conv1d_td_lstm", "conv2d_td_lstm"]:
            X_test_proc = np.expand_dims(X_test_proc, axis=-1)
            if model_type == "conv2d_td_lstm":
                time_per_chunk = 7
                X_test_proc = reshape_for_2d_td_lstm(X_test_proc, time_per_chunk)

        evaluate_model(
            model_type=model_type,
            model_path=model_file,
            label_encoder=label_encoder,
            X_test=X_test_proc,
            y_test_enc=y_test_enc,
            results_dir=config.RESULTS_PATH
        )

if __name__ == "__main__":
    main()
