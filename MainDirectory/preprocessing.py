import numpy as np
import pandas as pd
import json
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#nltk.download('stopwords') # may need to download stopwords and punkt first
#nltk.download('punkt') 

data = []
with open("Sarcasm_Headlines_Dataset_v2.json", 'r') as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Removing stopwords
stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    '''
    function to perform the text preprocessing steps: converting to lowercase, 
    removing punctuation, special characters, numbers, tokenization, 
    expanding contractions and removing stopwords
    '''
    # To lowercase
    text = text.lower()
    # Remove punctuation, special characters
    text = text.replace('[^\w\s]', '')
    # Expand contractions 
    text = expand_contractions(text)
    # Remove numbers 
    text = re.sub(r'\d+', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # Join tokens back to a single string
    processed_text = ' '.join(tokens)

    return processed_text


def expand_contractions(text):
    '''
    Function to expand contractions in the text
    '''
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "didn't": "did not",
        "doesn't": "does not",
        "haven't": "have not",
        "hasn't": "has not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "isn't": "is not"
    }
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    expanded_text = pattern.sub(lambda match: contractions[match.group(0)], text)
    return expanded_text

def vectorizeText(text):
    tfidf_vectorizer = TfidfVectorizer()
    X_pred = tfidf_vectorizer.fit_transform(text)

    return X_pred

def preprocess_and_split_data():

    # Applying to 'headline' column
    # Processed headlines stored in new column (to preserve original text data)
    df['processed_headline'] = df['headline'].apply(preprocess_text)

    # LabelEncoder from sklearn.preprocessing module to encode target variable 'is_sarcastic'
    # From categorical values to numeric values, then stored in 'is_sarcastic_encoded'
    label_encoder = LabelEncoder()
    df['is_sarcastic_encoded'] = label_encoder.fit_transform(df['is_sarcastic'])

    # TF-IDF feature extraction --> Feature matrix X
    # Convert textual data in 'processed_headline' column into numerical feature matrix X
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['processed_headline'])
    y = df['is_sarcastic_encoded']    # each value 0 or 1 (class of sarcasm)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    print("processing and splitting done")

    return X_train, X_test, y_train, y_test

