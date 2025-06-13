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

def train_model(model_type, X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le, num_classes):
    logging.info(f"\n======== Training model: {model_type} ========")

    if  config.MONO and model_type in ["conv1d_td", "conv1d_td_lstm", "conv2d_td_lstm", "conv2d_td", "conv2d_td_lstm_bd"]:
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
    if model_type == "conv2d_td_lstm" or model_type == "conv2d_td" or model_type == "conv2d_td_lstm_bd":
        time_per_chunk = 7
        X_train = reshape_for_2d_td_lstm(X_train, time_per_chunk)
        X_val = reshape_for_2d_td_lstm(X_val, time_per_chunk)
        X_test = reshape_for_2d_td_lstm(X_test, time_per_chunk)


    model = ModelFactory.create(model_type, input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    save_prefix = os.path.join(config.RESULTS_PATH, model_type)
    callbacks = [
        ModelCheckpoint(save_prefix + "_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, verbose=1)
    ]

    try:
        history = model.fit(
            X_train, y_train_enc,
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
