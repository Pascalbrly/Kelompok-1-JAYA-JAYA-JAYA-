import pandas as pd
import numpy as np
import pickle
import re
import tensorflow as tf
import os
import joblib
import nltk
import sqlite3

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from, LazyJSONEncoder, LazyString

nltk.download('stopwords')

app = Flask(__name__)

app.json_encoder = LazyJSONEncoder

swagger_template = {
    "info": {
        "title": 'API documentation for ML and DL',
        "version": "1.0.1",
        "description": "API for sentiment prediction using keras NN, LSTM, and MLPClassifier models",
    },
}
swagger_config = {
    'headers': [],
    'specs': [
        {
            'endpoint': 'docs',
            'route': '/docs.json',
        }
    ],
    'static_url_path': '/flasgger_static',
    'swagger_ui': True,
    'specs_route': "/docs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)
sentiment = ['negative', 'neutral', 'positive']

list_stopword = stopwords.words('indonesian')
list_stopword.extend(["yg", "guna","dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'nya', 'ber', 'banget', 'kali'])
list_stopword = list(dict.fromkeys(list_stopword))
list_stopword = set(list_stopword)
bukan_stopword = {'baik', 'masalah', 'yakin', 'tidak', 'pantas', 'lebih'}
final_stopword = set([word for word in list_stopword if word not in bukan_stopword])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

alay_df = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alay_filter = dict(zip(alay_df[0], alay_df[1]))

def normalisasi_alay(text):
    return ' '.join(alay_filter.get(word, word) for word in text.split(' '))

def remove_stopwords(text):
    return [word for word in text.split() if word not in final_stopword]

def stemming(text):
    return [stemmer.stem(word) for word in text]

def words_to_sentence(list_words):
    return ' '.join(list_words)

def cleansing(text):
    text = re.sub(r'\\t|\\n|\\u', ' ', text) #Menghapus karakter khusus seperti tab, baris baru, karakter Unicode, dan backslash.
    text = re.sub(r"https?:[^\s]+", ' ', text)  # Menghapus http / https
    text = re.sub(r'(\b\w+)-\1\b', r'\1', text)
    text = re.sub(r'[\\x]+[a-z0-9]{2}', '', text)  # Menghapus karakter yang dimulai dengan '\x' diikuti oleh dua karakter huruf atau angka
    # text = re.sub(r'(\d+)', r' \1 ', text)  # Memisahkan angka dari teks
    text = re.sub(r'[^a-zA-Z]+', ' ', text)  # Menghapus karakter kecuali huruf, dan spasi
    text = re.sub(r'\brt\b|\buser\b', ' ', text) # Menghapus kata-kata 'rt' dan 'user'
    text = text.lower()
    return text

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model_nn = load_model('nn_model.keras')
model_lstm = load_model('lstm_model.keras')

with open('mlp.pkl', 'rb') as handle:
    model_mlp = pickle.load(handle)

vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return "Welcome to the model API!"

@swag_from('docs/nn.yaml', methods=['POST'])
@app.route('/nn', methods=['POST'])
def nn():
    original_text = request.form.get('text')
    text_cleaned = cleansing(original_text)
    text_normalized = normalisasi_alay(text_cleaned)
    text_wsw = remove_stopwords(text_normalized)
    text_stem = stemming(text_wsw)
    text = words_to_sentence(text_stem)
    print("Cleaned text for NN:" ,text)  # Debugging
    feature = tokenizer.texts_to_sequences([text])
    X = pad_sequences(feature, maxlen=55)
    prediction = model_nn.predict(X)
    # print("Prediction probabilities (NN):", prediction)  # Debugging
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "NN Prediction Result",
        'data': {
            'text': original_text,
            'cleaned_text': text,
            'sentiment': get_sentiment,
        }
    }
    return jsonify(json_response)

@swag_from('docs/lstm.yaml', methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text_cleaned = cleansing(original_text)
    text_normalized = normalisasi_alay(text_cleaned)
    text_wsw = remove_stopwords(text_normalized)
    text_stem = stemming(text_wsw)
    text = words_to_sentence(text_stem)
    print("Cleaned text for LSTM:" ,text)  # Debugging
    feature = tokenizer.texts_to_sequences([text])
    X = pad_sequences(feature, maxlen=64)
    prediction = model_lstm.predict(X)
    # print("Prediction probabilities (LSTM):" ,prediction)  # Debugging
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "LSTM Prediction Result",
        'data': {
            'text': original_text,
            'cleaned_text': text,
            'sentiment': get_sentiment,
        }
    }
    return jsonify(json_response)

@swag_from('docs/mlp.yaml', methods=['POST'])
@app.route('/mlp', methods=['POST'])
def mlp():
    original_text = request.form.get('text')
    text_cleaned = cleansing(original_text)
    text_normalized = normalisasi_alay(text_cleaned)
    text_wsw = remove_stopwords(text_normalized)
    text_stem = stemming(text_wsw)
    text = words_to_sentence(text_stem)
    print("Cleaned text for MLP:",text)  # Debugging
    X = vectorizer.transform([text]).toarray()
    prediction = model_mlp.predict(X)
    # print("Prediction probabilities (MLP):", prediction)  # Debugging
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "MLP Prediction Result",
        'data': {
            'text': original_text,
            'cleaned_text': text,
            'sentiment': get_sentiment,
        }
    }
    return jsonify(json_response)

# Endpoint untuk upload file dan analisis menggunakan NN
@swag_from('docs/upload_nn.yaml', methods=['POST'])
@app.route('/upload_nn', methods=['POST'])
def upload_nn():
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas dengan batasan 100 baris
    data = pd.read_csv(file, encoding='latin-1', nrows=100)

    # Identify the text column
    text_column = None
    for col in data.columns:
        if data[col].dtype == 'object':
            text_column = col
            break

    if not text_column:
        return jsonify({
            'status_code': 400,
            'description': 'No text column found in the uploaded file'
        })

    # Ambil teks yang akan diproses dalam format list
    texts = data[text_column].tolist()

    # List to hold processed data
    processed_data = []

    for text in texts:
        # Lakukan cleansing pada teks
        text_cleaned = cleansing(text)
        text_normalized = normalisasi_alay(text_cleaned)
        text_wsw = remove_stopwords(text_normalized)
        text_stem = stemming(text_wsw)
        text_final = words_to_sentence(text_stem)

        # Prediction
        feature = tokenizer.texts_to_sequences([text_final])
        X = pad_sequences(feature, maxlen=55)
        prediction = model_nn.predict(X)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        # Append to processed data list
        processed_data.append({
            'text': text,
            'cleaned_text': text_final,
            'sentiment': get_sentiment
        })

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': processed_data,
    }

    response_data = jsonify(json_response)
    return response_data




# # Endpoint untuk upload file dan analisis menggunakan LSTM
# @swag_from('docs/upload_lstm.yaml', methods=['POST'])
# @app.route('/upload_lstm', methods=['POST'])
# def upload_lstm():
#     # Upladed file
#     # Upladed file
#     file = request.files.getlist('file')[0]
    
#     try:
#         # Import file csv ke Pandas dengan optimisasi
#         data = pd.read_csv(file, usecols=['Tweet'], dtype={'Tweet': str}, encoding='latin-1')
        
#         # Ambil teks yang akan diproses dalam format list
#         texts = data['Tweet'].tolist()
        
#         cleaned_texts = []
#         for text in texts:
#             text_cleaned = cleansing(text)
#             text_normalized = normalisasi_alay(text_cleaned)
#             text_wsw = remove_stopwords(text_normalized)
#             text_stem = stemming(text_wsw)
#             text_final = words_to_sentence(text_stem)
#             cleaned_texts.append(text_final)
        
#         json_response = {
#             'status_code': 200,
#             'description': "Teks yang sudah diproses",
#             'data': cleaned_texts,
#         }

#         response_data = jsonify(json_response)
#         return response_data

#     except Exception as e:
#         json_response = {
#             'status_code': 500,
#             'description': "Error during file processing",
#             'error': str(e),
#         }
#         return jsonify(json_response)

# # Endpoint untuk upload file dan analisis menggunakan MLP
# @swag_from('docs/upload_mlp.yaml', methods=['POST'])
# @app.route('/upload_mlp', methods=['POST'])
# def upload_mlp():
#     # Upladed file
#     # Upladed file
#     file = request.files.getlist('file')[0]
    
#     try:
#         # Import file csv ke Pandas dengan optimisasi
#         data = pd.read_csv(file, usecols=['Tweet'], dtype={'Tweet': str}, encoding='latin-1')
        
#         # Ambil teks yang akan diproses dalam format list
#         texts = data['Tweet'].tolist()
        
#         cleaned_texts = []
#         for text in texts:
#             text_cleaned = cleansing(text)
#             text_normalized = normalisasi_alay(text_cleaned)
#             text_wsw = remove_stopwords(text_normalized)
#             text_stem = stemming(text_wsw)
#             text_final = words_to_sentence(text_stem)
#             cleaned_texts.append(text_final)
        
#         json_response = {
#             'status_code': 200,
#             'description': "Teks yang sudah diproses",
#             'data': cleaned_texts,
#         }

#         response_data = jsonify(json_response)
#         return response_data

#     except Exception as e:
#         json_response = {
#             'status_code': 500,
#             'description': "Error during file processing",
#             'error': str(e),
#         }
#         return jsonify(json_response)

if __name__ == '__main__':
    app.run()
