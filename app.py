from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
import re
import string
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

def load_model():
    try:
        with open('fake_review_detection_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text, rating=5.0, category='Home_and_Kitchen_5'):
    model_data = load_model()
    if not model_data:
        return None
    
    le_category = model_data['category_encoder']
    tfidf_vec = model_data['tfidf_vectorizer']
    
    text_length = len(text)
    word_count = len(text.split())
    exclamation_count = text.count('!')
    question_count = text.count('?')
    capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    
    cleaned_text = clean_text(text)
    stop_words = set(stopwords.words('english'))
    unique_words = len(set(word_tokenize(cleaned_text)) - stop_words)
    avg_word_length = np.mean([len(word) for word in word_tokenize(cleaned_text)]) if word_tokenize(cleaned_text) else 0
    
    try:
        category_encoded = le_category.transform([category])[0]
    except:
        category_encoded = 0
    
    numerical_features = np.array([
        rating, category_encoded, text_length, word_count, exclamation_count,
        question_count, capital_ratio, punctuation_count, sentiment_scores['compound'],
        sentiment_scores['pos'], sentiment_scores['neu'], sentiment_scores['neg'],
        unique_words, avg_word_length
    ]).reshape(1, -1)
    
    tfidf_features = tfidf_vec.transform([cleaned_text]).toarray()
    X_combined = np.hstack([numerical_features, tfidf_features])
    
    return X_combined

def predict_review(text, rating=5.0, category='Home_and_Kitchen_5'):
    model_data = load_model()
    if not model_data:
        return None, None, None
    
    model = model_data['model']
    le_label = model_data['label_encoder']
    
    features = extract_features(text, rating, category)
    if features is None:
        return None, None, None
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    result = le_label.inverse_transform([prediction])[0]
    confidence = max(probabilities)
    fake_probability = probabilities[1] if len(probabilities) > 1 else 0
    
    return result, confidence, fake_probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() if request.is_json else request.form
        
        review_text = data.get('review_text', '')
        rating = float(data.get('rating', 5.0))
        category = data.get('category', 'Home_and_Kitchen_5')
        
        if not review_text.strip():
            return jsonify({'error': 'Please enter a review text'})
        
        result, confidence, fake_prob = predict_review(review_text, rating, category)
        
        if result is None:
            return jsonify({'error': 'Model not found. Please ensure the model file exists.'})
        
        response = {
            'prediction': 'Fake' if result == 'CG' else 'Real',
            'confidence': f'{confidence:.2%}',
            'fake_probability': f'{fake_prob:.2%}',
            'review_text': review_text,
            'rating': rating,
            'category': category
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'})
        
        df = pd.read_csv(file)
        results = []
        
        for index, row in df.iterrows():
            text = row.get('text_', row.get('review_text', ''))
            rating = row.get('rating', 5.0)
            category = row.get('category', 'Home_and_Kitchen_5')
            
            result, confidence, fake_prob = predict_review(text, rating, category)
            results.append({
                'index': index,
                'review': text[:100] + '...' if len(text) > 100 else text,
                'prediction': 'Fake' if result == 'CG' else 'Real',
                'confidence': f'{confidence:.2%}' if confidence else 'N/A',
                'fake_probability': f'{fake_prob:.2%}' if fake_prob else 'N/A'
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

if __name__ == '__main__':
    model_data = load_model()
    if model_data:
        print("Model loaded successfully!")
        print("Available categories:", list(model_data['category_encoder'].classes_))
    else:
        print("Warning: Model file not found. Please run the Jupyter notebook first to train and save the model.")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)