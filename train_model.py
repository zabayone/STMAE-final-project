import config
import logging
import gc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from utils import *
from models.model_factory import ModelFactory
from sklearn.metrics import classification_report
from data.data_augmenter import DataAugmenter
from data.feature_extractor import FeatureExtractor
from data.keras_data_generator import KerasAudioGenerator

def train_model(model_type, X_train_raw, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le, num_classes):
    logging.info(f"\n======== Training model: {model_type} ========")
    feature_extractor = FeatureExtractor(sample_rate=44100, bands=config.BANDS, mode=config.SIGNAL_PROCESSING_TYPE)

    print("Calcolo statistiche di normalizzazione sul training set...")
    # Estrai le feature SENZA augmentation per calcolare le statistiche
    initial_train_features = feature_extractor.extract(X_train_raw)
    initial_train_features_np = np.array(initial_train_features)

    # Calcola media e deviazione standard
    mean = np.mean(initial_train_features_np, axis=(0, 1))
    std = np.std(initial_train_features_np, axis=(0, 1)) + 1e-8

    # Salva le statistiche per un uso futuro (es. in eval.py)
    save_normalization_params(mean, std, os.path.join(config.RESULTS_PATH, "global_norm_stats.pkl"))

    # --- 2. PREPROCESSING DEI SET DI VALIDATION E TEST ---
    # Questi set vanno normalizzati e processati una sola volta
    print("Preprocessing validation data...")
    X_val = np.array(feature_extractor.extract(X_val))
    X_val = (X_val - mean) / std  # Normalizza

    print("Preprocessing test data...")
    X_test = np.array(feature_extractor.extract(X_test))
    X_test = (X_test - mean) / std
    # ... applica qui il reshape per X_val ...

    # data_augmenter = DataAugmenter(feature_extractor=feature_extractor, sample_rate=44100)
    # X_train = np.array(data_augmenter.process_and_extract(X_train))
    # X_val = np.array(data_augmenter.process_and_extract(X_val))
    # X_test = np.array(data_augmenter.process_and_extract(X_test))
    #
    # X_train, X_val, X_test, mean, std = normalize_features(X_train, X_val, X_test)
    #
    # joblib.dump(le, "label_encoder.pkl")
    # save_normalization_params(mean, std, os.path.join(config.RESULTS_PATH, "global_norm_stats.pkl"))


    if  config.MONO and model_type in ["conv1d_td", "conv1d_td_lstm", "conv2d_td_lstm", "conv2d_td", "conv2d_td_lstm_bd"]:
        # X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
    if model_type == "conv2d_td_lstm" or model_type == "conv2d_td" or model_type == "conv2d_td_lstm_bd":
        time_per_chunk = 7
        # X_train = reshape_for_2d_td_lstm(X_train, time_per_chunk)
        X_val = reshape_for_2d_td_lstm(X_val, time_per_chunk)
        X_test = reshape_for_2d_td_lstm(X_test, time_per_chunk)

    # --- 3. CREAZIONE DEL GENERATORE PER IL TRAINING ---
    data_augmenter = DataAugmenter(feature_extractor, 44100)
    train_generator = KerasAudioGenerator(
        x_set=X_train_raw,
        y_set=y_train_enc,
        batch_size=config.BATCH_SIZE,
        data_augmenter=data_augmenter,
        mean=mean,  # Passa le statistiche
        std=std,
        model_type=model_type,  # Passa il tipo di modello
        is_mono=config.MONO,
        time_per_chunk=7  # Esempio
    )


    model = ModelFactory.create(model_type, input_shape=X_val.shape[1:], num_classes=num_classes)
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    save_prefix = os.path.join(config.RESULTS_PATH, model_type)
    callbacks = [
        ModelCheckpoint(save_prefix + "_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=15, verbose=1)
    ]

    try:
        history = model.fit(
            train_generator,
            validation_data=(X_val, y_val_enc),
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            callbacks=callbacks,
            verbose=2
        )

        plot_training(history, save_prefix)
        joblib.dump(history.history, save_prefix + "_history.pkl")
        joblib.dump(le, "label_encoder.pkl")

        best_val_acc = max(history.history['val_accuracy'])
        print(f"Best validation accuracy: {best_val_acc:.4f}")

        best_model = load_model(save_prefix + "_best.h5")
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test_enc, verbose=0)
        print(f"Test accuracy: {test_accuracy:.4f}")

        y_pred = np.argmax(best_model.predict(X_test), axis=1)
        print("\nClassification Report:")
        print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
        plot_confusion_matrix(y_test_enc, y_pred, le.classes_, save_prefix + "_confusion_matrix.png")

    finally:
        K.clear_session()
        gc.collect()
