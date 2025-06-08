import tensorflow as tf
from models.base_model import BaseModel

class Conv1DTimeDistributedLSTMModel(BaseModel):
    def build(self, input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(None, input_shape[1], input_shape[2])))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(16, kernel_size=9, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(32, kernel_size=6, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

        model.add(tf.keras.layers.LSTM(64, return_sequences=True))

        model.add(tf.keras.layers.GlobalAveragePooling1D())

        model.add(tf.keras.layers.Dropout(0.35))

        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.summary()
        return model
