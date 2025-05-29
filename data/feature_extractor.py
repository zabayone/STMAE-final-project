import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=44100, n_mfcc=64):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract(self, audio_data):
        features = []
        for y in audio_data:
            mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            features.append(mfcc.T)  # T per avere (time, features)
        return features
