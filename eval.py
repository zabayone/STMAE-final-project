import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
from utils import reshape_for_2d_td_lstm, plot_confusion_matrix, apply_normalization
import config
from data.data_loader import DataLoader
from data.feature_extractor import FeatureExtractor

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

    plot_confusion_matrix(y_test_enc, y_pred, label_encoder.classes_,
                          os.path.join(results_dir, f"{model_type}_confusion_matrix.png"))


def main():
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    print("Caricamento dataset di test (non segmentato)...")
    loader = DataLoader(config.EVAL_DATASET_PATH, mono=config.MONO, slices_per_file=10)

    X_test_raw, y_test = loader.load(segment=False)

    print("Caricamento LabelEncoder...")
    label_encoder = joblib.load("label_encoder.pkl")
    y_test_enc = label_encoder.transform(y_test)

    print("Segmentazione in 10 chunk ciascuno...")
    slices_per_file = 10
    X_test_segments = []
    y_segments = []

    for i, full_feat in enumerate(X_test_raw):
        time_steps = full_feat.shape[-1]
        chunk_size = time_steps // slices_per_file

        for j in range(slices_per_file):
            start = j * chunk_size
            end = start + chunk_size
            if config.MONO:
                segment = full_feat[start:end]
            else:
                segment = full_feat[:, start:end]
            if segment.shape[-1] == chunk_size:
                X_test_segments.append(segment)
                y_segments.append(y_test_enc[i])

    X_test_segments = np.array(X_test_segments)

    print("Estrazione feature per ogni file...")
    extractor = FeatureExtractor(sample_rate=44100, bands=config.BANDS, mode=config.SIGNAL_PROCESSING_TYPE)

    X_test_segments = extractor.extract(X_test_segments)

    X_test_segments = np.array(X_test_segments)


    print("Caricamento parametri di normalizzazione...")
    norm_stats_path = os.path.join(config.RESULTS_PATH, "global_norm_stats.pkl")
    norm_stats = joblib.load(norm_stats_path)
    mean = norm_stats["mean"]
    std = norm_stats["std"]

    X_test_segments = apply_normalization(X_test_segments, mean, std)

    MODELS_TO_EVALUATE = [
        "conv1d",
        "conv1d_td",
        "conv1d_td_lstm",
        "conv2d_td_lstm",
        "conv2d_td",
        "conv2d",
        "conv2d_td_lstm_bd"
    ]

    for model_type in MODELS_TO_EVALUATE:
        print(f"\n--- Modello: {model_type} ---")

        model_file = os.path.join(config.RESULTS_PATH, model_type + "_best.h5")
        if not os.path.exists(model_file):
            print(f"⚠️  Modello non trovato: {model_file}")
            continue

        X_proc = X_test_segments.copy()
        if config.MONO and model_type in ["conv1d_td", "conv1d_td_lstm", "conv2d_td_lstm", "conv2d_td", "conv2d_td_lstm_bd"]:
            X_proc = np.expand_dims(X_proc, axis=-1)
        if model_type == "conv2d_td_lstm" or model_type == "conv2d_td" or model_type == "conv2d_td_lstm_bd":
            time_per_chunk = 7
            X_proc = reshape_for_2d_td_lstm(X_proc, time_per_chunk)

        print("Predizione su tutti i segmenti...")
        model = load_model(model_file)
        preds = model.predict(X_proc)


        num_files = len(X_test_raw)
        preds_per_file = preds.reshape(num_files, slices_per_file, -1)
        preds_mean = preds_per_file.mean(axis=1)

        y_pred = np.argmax(preds_mean, axis=1)

        acc = accuracy_score(y_test_enc, y_pred)
        print(f"Accuracy media per file: {acc:.4f}")

        report = classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_, digits=4)
        print(report)

        with open(os.path.join(config.RESULTS_PATH, f"{model_type}_report_per_file.txt"), "w") as f:
            f.write(f"Accuracy media per file: {acc:.4f}\n\n")
            f.write(report)

        plot_confusion_matrix(y_test_enc, y_pred, label_encoder.classes_,
                              os.path.join(config.RESULTS_PATH, f"{model_type}_confusion_matrix_per_file.png"))


if __name__ == "__main__":
    main()
