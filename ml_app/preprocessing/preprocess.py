import pandas as pd
import numpy as np
import tensorflow as tf
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import classification_report
from fuzzywuzzy import process
import re
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Normalization
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras import models 
from tensorflow import keras


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text.strip())


def eliminate_double_characters(text):
    # Initialize an empty string to store the processed text
    processed_text = ""
    
    # Iterate through the characters of the input text
    i = 0
    while i < len(text):
        # Append the current character to the processed text
        processed_text += text[i]
        
        # Check if the current character is repeated consecutively
        if i + 1 < len(text) and text[i] == text[i + 1]:
            # Move to the next character skipping the consecutive duplicates
            while i + 1 < len(text) and text[i] == text[i + 1]:
                i += 1
        # Move to the next character
        i += 1
    
    return processed_text


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def closest_word(text, dataset_texts):
    highest_similarity_score = 0
    closest_match = None
    for word in text.split():
        closest_match_for_word, similarity_score = process.extractOne(word, dataset_texts)
        if similarity_score > highest_similarity_score:
            highest_similarity_score = similarity_score
            closest_match = closest_match_for_word
    return closest_match, highest_similarity_score

def preprocess(df, dataset_texts):
    df['deskripsi'] = df['deskripsi'].apply(remove_extra_spaces)
    df['deskripsi'] = df['deskripsi'].apply(normalize_text)
    df['deskripsi'] = df['deskripsi'].apply(eliminate_double_characters)
    
    closest_words = []
    closest_words_num = []
    closest_words_score = []
    for sentence in df['deskripsi']:
        closest_match, similarity_score = closest_word(sentence, dataset_texts)
        if similarity_score < 20:
            closest_word_num = 0
            closest_match = "missing"
        else:
            closest_word_num = dataset_texts.index(closest_match) + 1
        closest_words.append(closest_match)
        closest_words_num.append(closest_word_num)
        closest_words_score.append(similarity_score)
    
    df['closest_words'] = closest_words
    df['closest_words_num'] = closest_words_num
    df['score'] = closest_words_score
    
    return df


def preprocess_input(deskripsi, nominal, dataset_texts):
    # Normalize deskripsi
    deskripsi = remove_extra_spaces(deskripsi)
    deskripsi = normalize_text(deskripsi)
    
    # Find the closest word and its number
    closest_match, _ = closest_word(deskripsi, dataset_texts)
    closest_word_num = dataset_texts.index(closest_match) + 1 if closest_match else 0
    
    # Normalize nominal
    nominal = nominal   # Assuming your training data was also scaled
    
    return np.array([[closest_word_num, nominal]], dtype=np.float32)