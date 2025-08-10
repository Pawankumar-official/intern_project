import re
import nltk 
import nltk
nltk.download('punkt_tab') 
import os

# Replace this with your actual path
label_dir = "labels"  # Folder where your .txt files are stored

documents = []
filenames = []

# Iterate through all .txt files
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        path = os.path.join(label_dir, filename)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if content:  # Avoid empty files
                documents.append(content)
                filenames.append(filename)

# Debug print to confirm
print(f"Loaded {len(documents)} documents.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english') # Moved vectorizer to global scope

def rank_documents_by_topic(documents, topic):
    # Filter out empty documents
    documents = [doc for doc in documents if doc.strip()]

    if not documents:
        raise ValueError("No valid (non-empty) documents provided.")

    tfidf_matrix = vectorizer.fit_transform([topic] + documents)

    if tfidf_matrix.shape[0] < 2:
        raise ValueError("TF-IDF matrix has insufficient documents to compare.")

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]
    return ranked_indices,cosine_sim   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english') # Moved vectorizer to global scope


def rank_documents_by_topic(documents, topic):
    # Filter out empty documents
    documents = [doc for doc in documents if doc.strip()]

    if not documents:
        raise ValueError("No valid (non-empty) documents provided.")

    tfidf_matrix = vectorizer.fit_transform([topic] + documents)

    if tfidf_matrix.shape[0] < 2:
        raise ValueError("TF-IDF matrix has insufficient documents to compare.")

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_indices = cosine_sim.argsort()[::-1]
    return ranked_indices,cosine_sim    
import nltk
nltk.download('punkt')
nltk.download('stopwords')   
import re
import nltk
import heapq

def summarize_text(text, num_sentences=5):
    # Ensure punkt tokenizer is loaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Ensure punkt_tab is loaded
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # Ensure stopwords are loaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Clean and tokenize
    clean_text = re.sub('[^a-zA-Z]', ' ', text.lower())
    clean_text = re.sub('\s+', ' ', clean_text)

    sentence_list = nltk.sent_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')

    # Frequency table
    word_frequencies = {}
    for word in nltk.word_tokenize(clean_text):
        if word not in stopwords:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    maximum_frequency = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= maximum_frequency

    # Score sentences
    sentence_scores = {}
    for sentence in sentence_list:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies and len(sentence.split()) < 30:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]

    # Top sentences
    top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    # Format nicely
    formatted_summary = "\n\n".join([f"• {sentence.strip()}" for sentence in top_sentences])
    return formatted_summary
def generate_combined_summary(topic, documents, filenames, top_n=5):
    ranked_indices, similarities = rank_documents_by_topic(documents, topic)
    print(f"🔍 Topic: {topic}\n")
    print("📄 Top Relevant Document Summaries:\n")

    final_summary = ""
    for i in range(top_n):
        idx = ranked_indices[i]
        print(f"{i+1}. 📘 {filenames[idx]} (Similarity: {similarities[idx]:.2f})")
        summary = summarize_text(documents[idx])
        print(summary)
        print("\n" + "-"*80 + "\n")
        final_summary += summary + " "

    print("\n📝 Combined Summary:\n")
    print(final_summary.strip())  
    
topic_query = "Impact of politics on society"
generate_combined_summary(topic_query, documents, filenames)   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os # Added import for os

vectorizer = TfidfVectorizer(stop_words='english')

def generate_combined_summary(topic_query, image_dir, label_dir):
    best_score = -1
    best_image_path = None
    best_label_text = ""

    label_texts = []
    label_filenames = []
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                label_texts.append(f.read().strip())
                label_filenames.append(label_file)

    if not label_texts:
        print("No label files found in the specified directory.")
        return {
            "summary": "",
            "image_path": None,
            "similarity": -1
        }

    # Fit the vectorizer on the topic and all label texts
    corpus = [topic_query] + label_texts
    tfidf_matrix = vectorizer.fit_transform(corpus)

    topic_embedding = tfidf_matrix[0:1]

    for i, label_text in enumerate(label_texts):
        label_embedding = tfidf_matrix[i+1:i+2] # Get the embedding for the current label
        similarity = cosine_similarity(topic_embedding, label_embedding).item()

        if similarity > best_score:
            best_score = similarity
            best_image_path = os.path.join(image_dir, os.path.splitext(label_filenames[i])[0] + ".jpg")
            best_label_text = label_text

    # Display best-matching image
    if best_image_path and os.path.exists(best_image_path):
        print(f"\n✅ Best Match:\n→ Image: {best_image_path}\n→ Similarity Score: {best_score:.4f}")
        # Display image using PIL (Pillow)
        try:
            img = Image.open(best_image_path)
            print(img) # Use display() for Colab
        except Exception as e:
            print(f"Could not display image: {e}")
    elif best_image_path:
        print(f"\n✅ Best Match found, but image file not found:\n→ Expected Image Path: {best_image_path}\n→ Similarity Score: {best_score:.4f}")
        print("→ Label Text:")
        print(best_label_text)
    else:
        print("No relevant documents found.")


    return {
        "summary": best_label_text,
        "image_path": best_image_path,
        "similarity": best_score
    }    
    result = generate_combined_summary(
    topic_query="Natural Environment",
    image_dir="/content/drive/MyDrive/Testing/Imges for testing",
    label_dir="/content/drive/MyDrive/Testing/Labels for testing"
)

    print("📄 Generated Summary:")
    print(result["summary"])
