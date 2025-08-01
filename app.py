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

load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Configuration & Data Loading ---
DATA_DIR = 'app_data'
FAISS_INDEX_PATH = os.path.join(DATA_DIR, 'book_indexb.faiss')
DATA_PATH = os.path.join(DATA_DIR, 'book_datab.pkl')

print("Loading data...")
try:
    index = faiss.read_index(FAISS_INDEX_PATH) 
    df = pd.read_pickle(DATA_PATH)
    
    if 'Title' not in df.columns:
        print("FATAL ERROR: 'Title' column not found in book_data.pkl.")
        exit()
    df['Title'] = df['Title'].astype(str)
    
    if len(df) != index.ntotal:
        print("FATAL ERROR: Data files are out of sync. Please regenerate your data files.")
        exit()

    title_to_idx = pd.Series(df.index, index=df['Title'])
    
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find a required file. {e}")
    print("Please ensure 'book_index.faiss' and 'book_data.pkl' are in the 'app_data' directory.")
    exit()

def string_similarity(a, b):
    """Calculates the similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def fetch_details_from_gemini(book_title):
    """Helper function to get book details from the Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {'error': 'Model is not configured on the server.'}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    prompt = f"""For the book titled "{book_title}", provide a JSON object with "author", "category", and a "description" of about 30 words. Only output the raw JSON."""
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}
    
    try:
        api_response = requests.post(url, headers=headers, json=payload, timeout=20)
        api_response.raise_for_status()
        result_json = api_response.json()
        raw_text = result_json['candidates'][0]['content']['parts'][0]['text']
        
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found in response", raw_text, 0)

        json_string = match.group(0)
        book_details = json.loads(json_string)

        return {
            'title': book_title,
            'author': book_details.get('author', 'N/A'),
            'category': book_details.get('category', 'N/A'),
            'description': book_details.get('description', 'N/A')
        }
    except requests.exceptions.RequestException:
        return {'error': 'Failed to communicate with the model.'}
    except (KeyError, IndexError, json.JSONDecodeError):
        return {'error': 'The model gave an unexpected response.'}

# --- API Endpoints ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '').lower()
        if not user_input:
            return jsonify({'error': 'Please enter a book title.'}), 400

        # Detail-fetching logic
        detail_keywords = ["tell me more about", "details for", "summary of", "what is"]
        for keyword in detail_keywords:
            if user_input.startswith(keyword):
                book_title = user_input.replace(keyword, "").strip().strip('"')
                details = fetch_details_from_gemini(book_title)
                if 'error' in details:
                    return jsonify(details), 500
                return jsonify({'type': 'gemini_details', 'details': details})

        all_titles = title_to_idx.index
        
        if all_titles.empty:
            return jsonify({'error': "I'm sorry, but my book database appears to be empty. I can't provide recommendations right now."}), 500

        best_match_title = max(all_titles, key=lambda title: string_similarity(user_input, title.lower()))
        
        match_score = string_similarity(user_input, best_match_title.lower())
        if match_score < 0.6:
            return jsonify({'error': f"Sorry, I couldn't find a book closely matching '{user_input}'."})

        query_idx_series = title_to_idx.get(best_match_title)
        if query_idx_series is None:
             return jsonify({'error': f"Could not find index for '{best_match_title}'."}), 404
        
        query_idx = int(query_idx_series.iloc[0] if isinstance(query_idx_series, pd.Series) else query_idx_series)

        query_embedding = index.reconstruct(query_idx).reshape(1, -1)
        
        k = 15
        distances, indices = index.search(query_embedding, k)
        
        recommended_df = df.iloc[indices[0]]
        
        final_recommendations = []
        stop_words = {'a', 'an', 'the', 'to', 'in', 'of', 'on', 'for', 'with', 'is', 'by', 'and', 'series'}
        query_keywords = {word.strip(string.punctuation) for word in best_match_title.lower().split() if word.strip(string.punctuation) not in stop_words and len(word.strip(string.punctuation)) > 2}

        for _, row in recommended_df.iterrows():
            if len(final_recommendations) >= 3:
                break
            if row['Title'] == best_match_title:
                continue
            
            recommended_title_words = {word.strip(string.punctuation) for word in row['Title'].lower().split()}
            if any(keyword in recommended_title_words for keyword in query_keywords):
                continue

            book_object = {"title": row.get('Title', 'N/A')}
            if book_object not in final_recommendations:
                final_recommendations.append(book_object)
                
        response = {
            'type': 'title_recommendation',
            'response_title': f"Based on \"{best_match_title.title()}\", you might also like:",
            'recommendations': final_recommendations
        }
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred in /chat endpoint: {e}")
        return jsonify({'error': 'An unexpected server error occurred.'}), 500

@app.route('/get_book_details', methods=['POST'])
def get_book_details():
    """Endpoint specifically for fetching details of a single book."""
    data = request.get_json()
    book_title = data.get('book_title')
    if not book_title:
        return jsonify({'error': 'Book title is required.'}), 400
    
    details = fetch_details_from_gemini(book_title)
    if 'error' in details:
        return jsonify(details), 500
    return jsonify(details)

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5001))
    app.run(host=host, port=port, debug=False)
