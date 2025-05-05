import pandas as pd
import textstat
import language_tool_python
import nltk
from tqdm import tqdm
import os

# Ensure NLTK components are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize grammar checker
tool = language_tool_python.LanguageTool('en-US')

def count_grammar_errors(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    matches = tool.check(text)
    return len(matches)


def avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return 0
    words = nltk.word_tokenize(text)
    return len(words) / len(sentences)

def pos_counts(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    counts = {
        'nouns': 0,
        'verbs': 0,
        'adjectives': 0,
        'adverbs': 0
    }
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            counts['nouns'] += 1
        elif tag.startswith('VB'):
            counts['verbs'] += 1
        elif tag.startswith('JJ'):
            counts['adjectives'] += 1
        elif tag.startswith('RB'):
            counts['adverbs'] += 1
    return counts

def extract_features(df):
    features = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['transcript']

        # 👇 Skip invalid or empty entries early
        if not isinstance(text, str) or not text.strip():
            continue

        grammar_errors = count_grammar_errors(text)
        num_words = len(text.split())
        num_sentences = text.count('.') + text.count('!') + text.count('?')
        flesch = textstat.flesch_reading_ease(text)
        gunning_fog = textstat.gunning_fog(text)
        smog = textstat.smog_index(text)
        pos_counts = get_pos_counts(text)

        features.append({
            'filename': row['filename'],
            'num_words': num_words,
            'num_sentences': num_sentences,
            'grammar_errors': grammar_errors,
            'flesch_reading_ease': flesch,
            'gunning_fog': gunning_fog,
            'smog_index': smog,
            **pos_counts
        })

    return pd.DataFrame(features)
