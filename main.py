
import logging
from data.preprocessing import prepare_datasets
import tensorflow as tf
from train_model import train_model

logging.basicConfig(level=logging.INFO)

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPU trovata: {gpus}")
        tf.config.set_visible_devices(gpus[0], 'GPU')
    else:
        logging.warning("Nessuna GPU trovata: il training userà la CPU.")

    X_train, y_train, X_val, y_val, X_test, y_test, le, num_classes = prepare_datasets()

    model_types = [

        "conv1d",
        "conv1d_td",
        "conv1d_td_lstm",
        "conv2d",
        "conv2d_td",
        "conv2d_td_lstm",
        "conv2d_td_lstm_bd"
    ]

    for model_type in model_types:
        train_model(model_type, X_train, y_train, X_val, y_val, X_test, y_test, le, num_classes)


if __name__ == "__main__":
    main()



