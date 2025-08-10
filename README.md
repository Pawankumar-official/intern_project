Natural Language Generation for Content Creation
📌 Overview
This project is a Flask-based web application that leverages Natural Language Processing (NLP) techniques to automatically generate summaries and find relevant images or documents for a given topic.
It supports two modes:

Image + Label Mode – Matches a topic with the most relevant image and generates a summary from its associated label text.

Text Document Mode – Finds and summarizes the most relevant document for the topic.

The system uses TF-IDF for semantic relevance ranking and extractive summarization for content generation.

🚀 Features
Topic-based document ranking using cosine similarity.

Automatic text summarization with NLTK.

Image retrieval based on label text similarity.

Flask API endpoints for integration with a frontend.

CORS enabled for cross-origin requests.

🛠 Tech Stack
Backend: Python, Flask

Libraries:

nltk – Tokenization, stopword removal, summarization

scikit-learn – TF-IDF vectorization, cosine similarity

Pillow – Image handling

Frontend: Static HTML/CSS/JS (served from Flask’s static folder)
