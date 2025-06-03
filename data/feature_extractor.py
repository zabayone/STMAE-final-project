import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=44100, bands=64, mode="mfcc"):
        self.sample_rate = sample_rate
        self.bands = bands
        self.mode = mode

    def extract(self, audio_data):
        features = []
        for y in audio_data:
            if self.mode == "mfcc":
                mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.bands)
                feat = mfcc.T

            elif self.mode == "melspec":
                mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.bands, power=2.0)
                log_mel = librosa.power_to_db(mel)
                feat = log_mel.T

            else:
                raise ValueError(f"Unknown feature mode: {self.mode}")

            features.append(feat)

        return features
