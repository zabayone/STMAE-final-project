
import tensorflow as tf

from models.base_model import BaseModel

class Conv1DModel(BaseModel):
    def build(self, input_shape, num_classes):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))

        model.add(tf.keras.layers.Conv1D(16, (9), activation='relu'))
        model.add(tf.keras.layers.MaxPool1D((2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(16, (3), activation='relu'))
        model.add(tf.keras.layers.MaxPool1D((2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv1D(32, (3), activation='relu'))
        model.add(tf.keras.layers.MaxPool1D((2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.summary()
        return model
