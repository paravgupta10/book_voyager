import os
import json
import string
import pickle
import requests
import numpy as np
import pandas as pd
import faiss
import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

load_dotenv()
app = Flask(__name__)
CORS(app)

model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
DATA_DIR = 'app_data'
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'book_index.faiss')
DATA_PATH = os.path.join(DATA_DIR, 'book_data.pkl')

print("Loading models and data...")
try:
    model = get_model()
    index = faiss.read_index(FAISS_INDEX_PATH)
    df = pd.read_pickle(DATA_PATH)
    title_to_idx = pd.Series(df.index, index=df['Title'])
    print("Models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Missing required file. {e}")
    exit()

def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def fetch_details_from_gemini(book_title):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {'error': 'GEMINI_API_KEY not found in environment variables.'}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    
    prompt = (
        f'For the book titled "{book_title}", provide a JSON object with exactly these fields: '
        '"author", "category", and "description" (40 words max). Output only raw JSON without explanation.'
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()

        raw_text = result['candidates'][0]['content']['parts'][0]['text']

        json_match = re.search(r'\{[\s\S]*?\}', raw_text)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in response", raw_text, 0)

        json_str = json_match.group(0)
        data = json.loads(json_str)

        if not all(k in data for k in ['author', 'category', 'description']):
            return {'error': 'Incomplete data from Gemini API.'}

        return data

    except (requests.exceptions.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        return {'error': f'Gemini API error or malformed response: {str(e)}'}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        if not user_input:
            return jsonify({'error': 'Please enter a book title.'}), 400

        detail_keywords = ["tell me more about", "details for", "summary of", "what is"]
        for keyword in detail_keywords:
            if user_input.lower().startswith(keyword):
                book_title = user_input[len(keyword):].strip().strip('"')
                details = fetch_details_from_gemini(book_title)
                if 'error' in details:
                    return jsonify(details), 500
                return jsonify({'type': 'gemini_details', 'details': details})

        all_titles = title_to_idx.index
        best_match_title = max(all_titles, key=lambda title: string_similarity(user_input.lower(), title.lower()))
        match_score = string_similarity(user_input.lower(), best_match_title.lower())
        if match_score < 0.6:
            return jsonify({'error': f"Sorry, no book closely matches '{user_input}'."})

        model = get_model()
        query_embedding = model.encode([best_match_title], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, 15)
        recommended_df = df.iloc[indices[0]]

        stop_words = {'a', 'an', 'the', 'to', 'in', 'of', 'on', 'for', 'with', 'is', 'by', 'and', 'series'}
        query_keywords = {word.strip(string.punctuation) for word in best_match_title.lower().split() if word.strip(string.punctuation) not in stop_words and len(word.strip(string.punctuation)) > 2}

        final_recommendations = []
        for _, row in recommended_df.iterrows():
            if len(final_recommendations) >= 3:
                break
            if row['Title'] == best_match_title:
                continue
            title_words = {word.strip(string.punctuation) for word in row['Title'].lower().split()}
            if query_keywords & title_words:
                continue
            book_object = {"title": row.get('Title', 'N/A')}
            if book_object not in final_recommendations:
                final_recommendations.append(book_object)

        return jsonify({
            'type': 'title_recommendation',
            'response_title': f'Based on "{best_match_title.upper()}", you might also like:',
            'recommendations': final_recommendations
        })

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': 'Unexpected server error.'}), 500

@app.route('/get_book_details', methods=['POST'])
def get_book_details():
    data = request.get_json()
    book_title = data.get('book_title')
    if not book_title:
        return jsonify({'error': 'Book title is required.'}), 400

    details = fetch_details_from_gemini(book_title)
    if 'error' in details:
        return jsonify(details), 500

    details['title'] = book_title

    return jsonify(details)


