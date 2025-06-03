import config
import logging
from data.data_loader import DataLoader
from data.feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from models.model_factory import ModelFactory
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

logging.basicConfig(level=logging.INFO)


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


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPU trovata: {gpus}")
    else:
        logging.warning("Nessuna GPU trovata: il training userà la CPU.")

    loader_train = DataLoader(config.DATASET_PATH, mono=config.MONO)
    X_all_raw, y_all = loader_train.load()

    loader_test = DataLoader(config.EVAL_DATASET_PATH, mono=config.MONO)
    X_test_raw, y_test = loader_test.load()

    logging.info(f"Totale esempi disponibili per il training: {len(X_all_raw)}")
    logging.info(f"Esempi test set: {len(X_test_raw)}")

    le = LabelEncoder()
    le.fit(y_all + y_test)  # Includi anche i label del test set per sicurezza
    y_all_enc = le.transform(y_all)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    # Split train/val (80/20 stratificato)
    X_train_raw, X_val_raw, y_train_enc, y_val_enc = train_test_split(
        X_all_raw, y_all_enc, test_size=0.2, stratify=y_all_enc, shuffle=True)

    logging.info("Estrazione feature...")
    extractor = FeatureExtractor(sample_rate=44100, bands=config.BANDS, mode=config.SIGNAL_PROCESSING_TYPE)
    X_train = np.array(extractor.extract(X_train_raw))
    X_val = np.array(extractor.extract(X_val_raw))
    X_test = np.array(extractor.extract(X_test_raw))
    X_train, X_val, X_test, _, _ = normalize_features(X_train, X_val, X_test)

    if config.MODEL_TYPE == "conv1d_td" or config.MODEL_TYPE == "conv1d_td_lstm" or config.MODEL_TYPE == "conv2d_td_lstm":
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        if config.MODEL_TYPE == "conv2d_td_lstm":
            time_per_chunk = 7
            X_train = reshape_for_2d_td_lstm(X_train, time_per_chunk)
            X_val = reshape_for_2d_td_lstm(X_val, time_per_chunk)
            X_test = reshape_for_2d_td_lstm(X_test, time_per_chunk)

    model = ModelFactory.create(config.MODEL_TYPE, input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer=Adam(1e-2), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    save_prefix = os.path.join(config.RESULTS_PATH, config.MODEL_TYPE)
    callbacks = [
        ModelCheckpoint(save_prefix + "_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, verbose=1)
    ]

    logging.info("Training...")
    try:
        history = model.fit(
            X_train, y_train_enc,
            validation_data=(X_val, y_val_enc),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=2
        )

        # Salva plot e history
        plot_training(history, save_prefix)
        joblib.dump(history.history, save_prefix + "_history.pkl")
        joblib.dump(le, "label_encoder.pkl")

        # Best val accuracy
        best_val_acc = max(history.history['val_accuracy'])
        print(f"Best validation accuracy: {best_val_acc:.4f}")

        # Valutazione sul test set con il modello migliore
        best_model = load_model(save_prefix + "_best.h5")
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test_enc, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Altre metriche
        y_pred = np.argmax(best_model.predict(X_test), axis=1)
        print("\nClassification Report:")
        print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
        plot_confusion_matrix(y_test_enc, y_pred, le.classes_, save_prefix + "_confusion_matrix.png")

    finally:
        K.clear_session()
        gc.collect()

def reshape_for_2d_td_lstm(X, time_per_chunk):
    # X shape: (samples, time_steps, freq_bins, 1)
    samples, time_steps, freq_bins, channels = X.shape
    assert time_steps % time_per_chunk == 0, "Il numero di time steps deve essere divisibile per time_per_chunk"

    time_chunks = time_steps // time_per_chunk

    # Reshape in: (samples, time_chunks, time_per_chunk, freq_bins, channels)
    X_new = X.reshape((samples, time_chunks, time_per_chunk, freq_bins, channels))
    return X_new

def normalize_features(X_train, X_val, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8  # Aggiungi epsilon per evitare divisione per zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


if __name__ == "__main__":
    main()



