# 📚 Book Voyager.AI

A smart chatbot that helps users:
- ✅ Get book recommendations based on similar descriptions using **Sentence Transformers + FAISS**
- ✅ Ask for **book details** (title, author, category, description) using fuzzy search from natural language queries

---

## 🚀 Features

- 🔍 **Semantic Search**: Recommend similar books by comparing description embeddings using FAISS.
- 💬 **Intelligent Q&A**: Extracts and matches user queries to book titles using fuzzy string matching.
- 🧠 **Natural Language Understanding**: Understands queries like:
  - “Tell me about Dune”
  - “Give me details of The Alchemist”
- ⚡ Fast and scalable on local CSV/database

---

## 🧱 Tech Stack

- Python
- [sentence-transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [pandas](https://pandas.pydata.org/)

---

## 🧠 How it Works

###  **Book Recommendations**
- Uses **sentence-transformers** to encode book descriptions.
- Builds a **FAISS** index for fast similarity search.
- Returns top-3 similar books for the user queried book.

