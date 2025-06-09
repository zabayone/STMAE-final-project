import tensorflow as tf
from models.base_model import BaseModel

class Conv2DTimeDistributedLSTMBDModel(BaseModel):
    def build(self, input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(None, input_shape[1], input_shape[2], input_shape[3])))

        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(16, (2, 3), activation='relu', padding='same')
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.SpatialDropout2D(0.2)
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((1, 2))
        ))


        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (2, 3), activation='relu', padding='same')
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.SpatialDropout2D(0.2)
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((1, 2))
        ))


        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(64, (2, 3), activation='relu', padding='same')
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.SpatialDropout2D(0.2)
        ))
        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((2, 2))
        ))


        model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        ))

        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ))
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        model.add(tf.keras.layers.Dropout(0.35))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.summary()
        return model
