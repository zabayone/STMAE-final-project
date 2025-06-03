import os
import librosa

class DataLoader:
    def __init__(self, dataset_path, sample_rate=44100, mono=True, slices_per_file=10):
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.slices_per_file = slices_per_file

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
                        chunks = self.split_into_chunks(y)

                        for chunk in chunks:
                            audio_data.append(chunk)
                            labels.append(label)


                    except Exception as e:
                        print(f"Errore nel file {path}: {e}")

        return audio_data, labels

    def split_into_chunks(self, y):
        total_samples = len(y)
        chunk_size = total_samples // self.slices_per_file

        chunks = []
        for i in range(self.slices_per_file):
            start = i * chunk_size
            end = start + chunk_size
            chunk = y[start:end]
            if len(chunk) == chunk_size:  # evita chunk incompleti
                chunks.append(chunk)
        return chunks