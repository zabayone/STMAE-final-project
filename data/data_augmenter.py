import numpy as np
import librosa
from tqdm import tqdm

class DataAugmenter:

    def __init__(self, feature_extractor, sample_rate=44100):

        if not hasattr(feature_extractor, 'extract'):
            raise ValueError("L'oggetto feature_extractor deve avere un metodo 'extract'")

        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

        self.pitch_shift_steps = 2
        self.time_stretch_rate_range = (0.8, 1.2)
        self.noise_snr_db_range = (5, 20)

    def _pitch_shift(self, waveform):

        steps = np.random.uniform(-self.pitch_shift_steps, self.pitch_shift_steps)
        return librosa.effects.pitch_shift(y=waveform, sr=self.sample_rate, n_steps=steps)

    def _time_stretch(self, waveform):

        original_length = len(waveform)
        rate = np.random.uniform(self.time_stretch_rate_range[0], self.time_stretch_rate_range[1])

        stretched_waveform = librosa.effects.time_stretch(y=waveform, rate=rate)

        current_length = len(stretched_waveform)

        if current_length > original_length:
            return stretched_waveform[:original_length]
        elif current_length < original_length:
            n_repeats = int(np.ceil(original_length / current_length))
            looped_waveform = np.tile(stretched_waveform, n_repeats)
            return looped_waveform[:original_length]
        else:
            # La lunghezza è già corretta
            return stretched_waveform

    def _add_noise(self, waveform):
        snr_db = np.random.uniform(self.noise_snr_db_range[0], self.noise_snr_db_range[1])
        signal_power = np.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
        return waveform + noise

    def process_and_extract(self, waveforms_list):
        augmented_waveforms = []

        # print(f"Applicazione augmentation su {len(waveforms_list)} file...")
        for waveform in waveforms_list:
            augmented_waveform = waveform.copy()

            # Applica il pitch shifting con probabilità 1/3
            if np.random.rand() < 1 / 3:
                augmented_waveform = self._pitch_shift(augmented_waveform)

            # Applica il time stretching con probabilità 1/3
            if np.random.rand() < 1 / 3:
                augmented_waveform = self._time_stretch(augmented_waveform)

            # Applica il rumore con probabilità 1/3
            if np.random.rand() < 1 / 3:
                augmented_waveform = self._add_noise(augmented_waveform)

            augmented_waveforms.append(augmented_waveform)


        return self.feature_extractor.extract(augmented_waveforms)