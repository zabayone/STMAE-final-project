import tensorflow as tf
from models.base_model import BaseModel

class Conv1DTimeDistributedModel(BaseModel):
    def build(self, input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))  # (time, freq, 1)

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(16, kernel_size=9, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPool1D(pool_size=2, padding='same')))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.BatchNormalization()))

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')))

        model.add(tf.keras.layers.GlobalAveragePooling1D())

        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.summary()
        return model
