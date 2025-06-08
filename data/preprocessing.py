import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data.data_loader import DataLoader
from data.feature_extractor import FeatureExtractor
from utils import normalize_features, save_normalization_params
import config
import joblib

def prepare_datasets():
    print("estrazione dataset train")
    loader_train = DataLoader(config.DATASET_PATH, mono=config.MONO)
    X_all_raw, y_all = loader_train.load()

    print("estrazione dataset test")
    loader_test = DataLoader(config.EVAL_DATASET_PATH, mono=config.MONO)
    X_test_raw, y_test = loader_test.load()

    le = LabelEncoder()
    le.fit(y_all + y_test)
    y_all_enc = le.transform(y_all)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    X_train_raw, X_val_raw, y_train_enc, y_val_enc = train_test_split(
        X_all_raw, y_all_enc, test_size=0.2, stratify=y_all_enc, shuffle=True)


    extractor = FeatureExtractor(sample_rate=44100, bands=config.BANDS, mode=config.SIGNAL_PROCESSING_TYPE)
    print("estrazione feature dataset train")
    X_train = np.array(extractor.extract(X_train_raw))
    print("estrazione feature dataset val")
    X_val = np.array(extractor.extract(X_val_raw))
    print("estrazione feature dataset test")
    X_test = np.array(extractor.extract(X_test_raw))
    X_train, X_val, X_test, mean, std = normalize_features(X_train, X_val, X_test)

    joblib.dump(le, "label_encoder.pkl")
    save_normalization_params(mean, std, os.path.join(config.RESULTS_PATH, "global_norm_stats.pkl"))

    return X_train, y_train_enc, X_val, y_val_enc, X_test, y_test_enc, le, num_classes
