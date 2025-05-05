import whisper
import os
import pandas as pd
from tqdm import tqdm

# Load Whisper model (you can switch 'base' to 'small' or 'medium' if needed)
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path, fp16=False)
        return result['text']
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return ""

def process_folder(folder_path, output_csv):
    data = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.wav'):
            full_path = os.path.join(folder_path, filename)
            transcript = transcribe_audio(full_path)
            data.append({'filename': filename, 'transcript': transcript})

    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"✅ Transcripts saved to {output_csv}")

if __name__ == "__main__":
    process_folder("dataset/audios_train", "train_transcripts.csv")
    process_folder("dataset/audios_test", "test_transcripts.csv")
