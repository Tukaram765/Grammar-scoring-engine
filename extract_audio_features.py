import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        features = {}

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(mfccs.shape[0]):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)

        # RMSE
        rmse = librosa.feature.rms(y=y)
        features['rmse_mean'] = np.mean(rmse)

        return features

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return {}

def process_audio_folder(folder_path):
    data = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.wav'):
            full_path = os.path.join(folder_path, filename)
            feats = extract_audio_features(full_path)
            feats['filename'] = filename
            data.append(feats)

    return pd.DataFrame(data)

if __name__ == "__main__":
    train_audio_df = process_audio_folder("dataset/audios_train")
    train_audio_df.to_csv("train_audio_features.csv", index=False)

    test_audio_df = process_audio_folder("dataset/audios_test")
    test_audio_df.to_csv("test_audio_features.csv", index=False)

    print("✅ Acoustic feature extraction complete!")
