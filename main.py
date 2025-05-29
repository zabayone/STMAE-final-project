import config
import logging
from data.data_loader import DataLoader
from data.feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from models.model_factory import ModelFactory
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt
import os


logging.basicConfig(level=logging.INFO)


def plot_training(history):
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
    plt.savefig(os.path.join(config.RESULTS_PATH, config.MODEL_TYPE+"training_history.png"))
    plt.close()


def main():
    loader_train = DataLoader(config.DATASET_PATH, mono=config.MONO)
    X_train_raw, y_train = loader_train.load()

    loader_val = DataLoader(config.EVAL_DATASET_PATH, mono=config.MONO)
    X_val_raw, y_val = loader_val.load()


    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    num_classes = len(le.classes_)


    logging.info("Estrazione feature...")
    extractor = FeatureExtractor(sample_rate=44100, n_mfcc=40)
    X_train_feat = extractor.extract(X_train_raw)
    X_val_feat = extractor.extract(X_val_raw)

    X_train = np.array(X_train_feat)
    X_val = np.array(X_val_feat)
    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_val_cat = to_categorical(y_val_enc, num_classes)

    if config.MODEL_TYPE == "conv1d_td":
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)

    model = ModelFactory.create(config.MODEL_TYPE, input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(os.path.join(config.MODEL_SAVE_PATH, config.MODEL_TYPE+"_best.h5"), monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, verbose=1)
    ]

    logging.info("Training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    plot_training(history)

    joblib.dump(le, "label_encoder.pkl")

if __name__ == "__main__":
    main()
