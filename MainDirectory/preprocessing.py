import numpy as np
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Downloading necessary NLTK resources
nltk.download('stopwords')


# Loading the dataset
# df = pd.read_json("C:/Users/Luisa/Documents/Uni/SoSe 2023/SciPy/Project/Sarcasm_Headlines_Dataset_v2.json") 
# --> did not work because of parsing error in json file
# read file line by line and append each JSON object to list
data = []
with open("C:/Users/Luisa/Documents/Uni/SoSe 2023/SciPy/Project/Sarcasm_Headlines_Dataset_v2.json", 'r') as file:
    for line in file:
        data.append(json.loads(line))

df = pd.DataFrame(data)



# Actual Text Preprocessing
# removing stopwords
stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    '''
    function to perform the text preprocessing steps: converting to lowercase, 
    removing punctuation and special characters, tokenization, and removing stopwords
    '''
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = text.replace('[^\w\s]', '')
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    # Join tokens back to a single string
    processed_text = ' '.join(tokens)

    return processed_text


# Applying to 'headline' column
# Processed headlines stored in new column
df['processed_headline'] = df['headline'].apply(preprocess_text)


# LabelEncoder from sklearn.preprocessing module to encode target variable 'is_sarcastic'
# from categorical values to numeric values
# Then stored in 'is_sarcastic_encoded'
label_encoder = LabelEncoder()
df['is_sarcastic_encoded'] = label_encoder.fit_transform(df['is_sarcastic'])


# TF-IDF feature extraction --> Feature matrix X
# converts textual data in 'processed_headline' column into numerical feature matrix X
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['processed_headline'])
y = df['is_sarcastic_encoded']    #value corresponds to the encoded class of sarcasm (0 or 1)


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# testing
print(df['processed_headline'].head())
