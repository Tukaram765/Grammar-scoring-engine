# 🗣️ Grammar Scoring Engine for Voice Samples

This project is part of SHL's Research Intern Take-Home Assignment. The goal is to build a system that evaluates the grammar quality of spoken language from audio clips and predicts a score between **0 to 5** based on a MOS Likert scale.

---

## 📁 Project Structure

. ├── dataset/ │ ├── train.csv │ ├── test.csv │ └── sample_submission.csv ├── train_audio_features.csv ├── test_audio_features.csv ├── train_transcripts.csv ├── test_transcripts.csv ├── train_combined.csv ├── test_combined.csv ├── submission.csv ├── train_model.py ├── grammar_model.pkl ├── scaler.pkl ├── GrammarScoringEngine.ipynb └── README.md


---

## 🧠 Objective

Given audio samples of spoken English, the system should:
- Extract useful **transcript-based** and **audio-based** features
- Train a regression model to predict **grammar scores**
- Evaluate the model using **Pearson Correlation** and **RMSE**
- Output predictions for the test set in `submission.csv`

---

## 📦 Dataset Description

- `train.csv`: Filenames and grammar score labels for training
- `test.csv`: Filenames and dummy labels for testing
- `sample_submission.csv`: Format for submission

---

## 🛠️ Features Used

- **Transcript features**: word count, average word length, filler words, etc.
- **Audio features**: MFCCs, pitch, intensity, speaking rate, pauses, etc.

---

## 🧪 Model

- **Model Used**: `RandomForestRegressor` from Scikit-learn
- **Scaling**: `StandardScaler` applied to feature set
- **Evaluation Metrics**:
  - ✅ Pearson Correlation (main metric)
  - 📉 RMSE (Root Mean Square Error)

---

## 📈 Results

| Metric            | Score     |
|-------------------|-----------|
| Pearson Correlation | 0.72 (example) |
| RMSE              | 0.48 (example)  |

> *(Update with actual results once evaluated)*

---

## 🚀 How to Run

1. Clone or download the repository
2. Ensure you have the following installed:
    ```bash
    pip install pandas scikit-learn scipy joblib
    ```
3. Run the model training script:
    ```bash
    python train_model.py
    ```
4. The predictions will be saved as `submission.csv`

---

## 📊 Visualization & Interpretability

Basic model evaluation and interpretability included via:
- Pearson correlation
- Feature importance plots (can be added in notebook)

---

## 📌 Notes

- Only open-source tools used
- Clean, interpretable model with explainable predictions
- All preprocessing, model training, and inference steps included

---

## 👨‍💻 Author

- **Your Name**
- For: SHL Research Internship Assignment

---

## 📄 License

This project is for academic/research demonstration. Not for commercial use.

