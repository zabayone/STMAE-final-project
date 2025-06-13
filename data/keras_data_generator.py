import numpy as np
import tensorflow as tf
from .data_augmenter import DataAugmenter


class KerasAudioGenerator(tf.keras.utils.Sequence):


    def __init__(self, x_set, y_set, batch_size, data_augmenter,
                 mean, std, model_type, is_mono, time_per_chunk=7, shuffle=True):
        """
        Inizializza il generatore.

        Args:
            x_set (list): La lista di waveform (es. X_train_raw).
            y_set (np.array): La lista di etichette numeriche (es. y_train_enc).
            batch_size (int): La dimensione del batch.
            data_augmenter (DataAugmenter): Un'istanza della classe DataAugmenter.
            mean (np.array): La media pre-calcolata per la normalizzazione.
            std (np.array): La deviazione standard pre-calcolata per la normalizzazione.
            model_type (str): Il tipo di modello (es. "conv2d_td_lstm") per il reshaping.
            is_mono (bool): Flag per sapere se l'audio è mono.
            time_per_chunk (int): Parametro per il reshape dei modelli TD.
            shuffle (bool): Se mescolare i dati alla fine di ogni epoca.
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.data_augmenter = data_augmenter
        self.mean = mean
        self.std = std
        self.model_type = model_type
        self.is_mono = is_mono
        self.time_per_chunk = time_per_chunk
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        """Restituisce il numero di batch per epoca."""
        return int(np.floor(len(self.x) / self.batch_size))

    def _reshape_for_2d_td_lstm(self, X_batch):
        """Funzione helper per il reshape dei modelli TD 2D."""
        # X_batch shape: (batch_size, time_steps, freq_bins, 1)
        samples, time_steps, freq_bins, channels = X_batch.shape
        if time_steps % self.time_per_chunk != 0:
            raise ValueError(
                f"Il numero di time steps ({time_steps}) deve essere divisibile per time_per_chunk ({self.time_per_chunk})")

        time_chunks = time_steps // self.time_per_chunk
        # Reshape in: (batch_size, time_chunks, time_per_chunk, freq_bins, channels)
        X_new = X_batch.reshape((samples, time_chunks, self.time_per_chunk, freq_bins, channels))
        return X_new

    def __getitem__(self, index):
        """Genera un batch di dati."""
        # 1. Prende gli indici per il batch corrente
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # 2. Prende le waveform e le etichette per quegli indici
        batch_x_waveforms = [self.x[k] for k in batch_indices]
        batch_y = self.y[batch_indices]

        # 3. Usa DataAugmenter per processare SOLO il batch corrente
        batch_x_features = self.data_augmenter.process_and_extract(batch_x_waveforms)
        batch_x_features = np.array(batch_x_features)

        # 4. Normalizza il batch usando le statistiche pre-calcolate
        batch_x_features = (batch_x_features - self.mean) / self.std

        # 5. Applica il reshaping finale in base al tipo di modello
        if self.is_mono and self.model_type in ["conv1d_td", "conv1d_td_lstm", "conv2d_td_lstm", "conv2d_td",
                                                "conv2d_td_lstm_bd"]:
            batch_x_features = np.expand_dims(batch_x_features, axis=-1)

        if self.model_type in ["conv2d_td_lstm", "conv2d_td", "conv2d_td_lstm_bd"]:
            batch_x_features = self._reshape_for_2d_td_lstm(batch_x_features)

        return batch_x_features, batch_y

    def on_epoch_end(self):
        """Mescola gli indici alla fine di ogni epoca."""
        if self.shuffle:
            np.random.shuffle(self.indices)
