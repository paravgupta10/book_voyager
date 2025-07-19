import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# --- Configuration ---
CSV_PATH = 'cleaned-book.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 128
OUTPUT_DIR = 'app_data'
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'book_index.faiss')
DATA_PATH = os.path.join(OUTPUT_DIR, 'book_data.pkl')

# --- Create output directory if it doesn't exist ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Step 1: Data Loading and Preparation ---
print("Step 1: Loading and preparing data...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: '{CSV_PATH}' not found. Please place it in the same directory.")
    exit()

# Clean up data: drop rows with missing titles as they are essential
df.dropna(subset=['Title'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Combine text fields into a single 'content' column for embedding
df['content'] = df['Title'].fillna('') + '. ' + df['Description'].fillna('') + ' Category: ' + df['Category'].fillna('')
corpus = df['content'].tolist()
print(f"Data loaded. Total entries to process: {len(corpus)}")

# --- Step 2: Initialize BERT Model ---
print(f"\nStep 2: Initializing Sentence Transformer model '{MODEL_NAME}'...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(MODEL_NAME, device=device)
print(f"Model loaded on device: {device}")
embedding_dim = model.get_sentence_embedding_dimension()

# --- Step 3: Generate Embeddings ---
print("\nStep 3: Generating embeddings... (This will take some time)")
corpus_embeddings = model.encode(corpus, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
print(f"Embeddings generated. Shape: {corpus_embeddings.shape}")

# --- Step 4: Build and Save Faiss Index ---
print("\nStep 4: Building and saving Faiss index...")
index = faiss.IndexFlatL2(embedding_dim)
index.add(corpus_embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"Faiss index built and saved to {FAISS_INDEX_PATH}")

# --- Step 5: Save Processed DataFrame ---
# We only need the 'Title' for lookup, but saving a bit more can be useful.
# Let's keep Title, Authors, and Description for display purposes.
processed_df = df[['Title', 'Authors', 'Description']].copy()
with open(DATA_PATH, 'wb') as f:
    pickle.dump(processed_df, f)
print(f"Processed book data saved to {DATA_PATH}")

print("\n--- Pre-computation complete! ---")
print("You can now run the Flask application.")
