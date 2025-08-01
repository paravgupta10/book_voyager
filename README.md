# 📚 Book Voyager.AI

A smart chatbot that provides book recommendations and details through a simple, intuitive interface.

**[➡️ View the Live Demo](https://book-voyager-frontend-jqgu.onrender.com/)**

---

## 🚀 Features

-   🔍 **Semantic Recommendations**: Enter a book title, and get recommendations for 3 books with similar themes and descriptions.
-   💬 **Natural Language Q&A**: Ask for details about a specific book, like "tell me more about Dune," to get its author, category and description.
-   ⚡ **Fast & Lightweight**: The live application is built to be fast and memory-efficient, ensuring a smooth user experience.
-   🌐 **Scalable**: The backend is designed to handle a large database of books from a simple CSV file.

---

## 🧱 Tech Stack

-   **Frontend**: React.js
-   **Backend**: Python (Flask)
-   **Deployment**: Render (Web Service + Static Site)
-   **AI & Data Processing**:
    -   **Embeddings**: `sentence-transformers`
    -   **Vector Search**: `faiss` (Facebook AI Similarity Search)
    -   **Data Handling**: `pandas`

---

## 🧠 How it Works

The project is split into two main parts: an offline preprocessing step and the live application.

### 1. Preprocessing (Offline)

-   A Python script (`data_processor.py`) reads a CSV file of books.
-   It uses a `sentence-transformers` model to convert each book's title and description into a numerical vector (embedding).
-   These embeddings are stored in a highly-efficient, **FAISS index** (`book_indexb.faiss`) for incredibly fast similarity lookups.

### 2. Live Application (Online)

-   The lightweight Flask backend loads the pre-computed FAISS index and book data. 
-   When a user enters a book title:
    1.  The app finds the closest matching title in its database.
    2.  It retrieves the pre-made embedding for that book from the FAISS index using `index.reconstruct()`.
    3.  It uses that embedding to search the entire index for the most similar books.
-   If a user asks for details, the app queries the Gemini API to provide a real-time summary.

---

