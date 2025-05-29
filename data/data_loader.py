import os
import librosa

class DataLoader:
    def __init__(self, dataset_path, sample_rate=44100, mono=True):
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self):
        audio_data = []
        labels = []

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    path = os.path.join(root, file)
                    label = os.path.basename(root)
                    try:
                        y, _ = librosa.load(path, sr=self.sample_rate, mono=self.mono)
                        audio_data.append(y)
                        labels.append(label)
                    except Exception as e:
                        print(f"Errore nel file {path}: {e}")

        return audio_data, labels
