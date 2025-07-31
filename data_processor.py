import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
import time


CSV_PATH = 'book-precompute.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 256 
OUTPUT_DIR = 'app_data'
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'book_index.faiss')
DATA_PATH = os.path.join(OUTPUT_DIR, 'book_data.pkl')

try:
    CSV_PATH
except NameError:
    raise ValueError("ERROR: The 'CSV_PATH' variable is not defined. Please define it before running this script.")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: '{OUTPUT_DIR}'")

print("\n[Step 1/5] Loading and preparing data...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded '{CSV_PATH}'.")
except FileNotFoundError:
    raise FileNotFoundError(f"FATAL ERROR: '{CSV_PATH}' not found. Please ensure the file was uploaded correctly.")

initial_rows = len(df)
df.dropna(subset=['Title'], inplace=True)
df.reset_index(drop=True, inplace=True)
if initial_rows > len(df):
    print(f"Removed {initial_rows - len(df)} rows with missing titles.")

df['content'] = df['Title'].fillna('') + '. ' + df['Description'].fillna('') + ' Category: ' + df['Category'].fillna('')
corpus = df['content'].tolist()
print(f"Data prepared. Total entries to process: {len(corpus)}")

print(f"\n[Step 2/5] Initializing model '{MODEL_NAME}'...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{'GPU detected' if device == 'cuda' else 'Warning: GPU not detected'}. Running on '{device}'.")

model = SentenceTransformer(MODEL_NAME, device=device)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Embedding dimension: {embedding_dim}")

print("\n[Step 3/5] Generating embeddings... ")
start_time = time.time()
corpus_embeddings = model.encode(
    corpus,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True
)
end_time = time.time()
print(f"Embeddings generated in {end_time - start_time:.2f} seconds.")
print(f"Shape of embeddings matrix: {corpus_embeddings.shape}")

print("\n[Step 4/5] Building and saving a compressed Faiss index...")

n_books = len(corpus_embeddings)
nlist = max(1, int(np.sqrt(n_books)))  # ✅ Avoid nlist=0
m = 8
nbits = 8

quantizer = faiss.IndexFlatL2(embedding_dim)
index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)

print(f"Training the index on {n_books} vectors...")
index.train(corpus_embeddings)

if not index.is_trained:
    raise RuntimeError("FAISS index training failed.")

print("Adding embeddings to the index...")
index.add(corpus_embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"Compressed Faiss index with {index.ntotal} vectors saved to '{FAISS_INDEX_PATH}'")

print("\n[Step 5/5] Saving processed book data...")
processed_df = df[['Title', 'Description', 'Category']].copy()
with open(DATA_PATH, 'wb') as f:
    pickle.dump(processed_df, f)
print(f"Book metadata saved to '{DATA_PATH}'")

