from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from backend_module import generate_combined_summary, summarize_text, rank_documents_by_topic
import os

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_summary():
    data = request.get_json()
    topic = data.get("topic", "")
    use_images = data.get("use_images", True)

    if not topic.strip():
        return jsonify({"summary": "❌ No topic provided.", "similarity": 0, "image_url": None})

    if use_images:
        # 🖼 Image + Label Mode
        image_dir = os.path.join("static", "images")
        label_dir = "labels"

        result = generate_combined_summary(topic, image_dir, label_dir)

        # Format image URL if path exists
        if result["image_path"]:
            relative_path = result["image_path"].replace("\\", "/").replace("static/", "")
            result["image_url"] = f"/static/{relative_path}"
        else:
            result["image_url"] = None

        return jsonify({
            "summary": result.get("summary", "⚠️ No summary generated."),
            "similarity": result.get("similarity", 0),
            "image_url": result.get("image_url")
        })

    else:
        # 📄 Text Document Mode
        doc_dir = "documents"
        documents, filenames = [], []

        for fname in os.listdir(doc_dir):
            if fname.endswith(".txt"):
                path = os.path.join(doc_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            documents.append(text)
                            filenames.append(fname)
                except Exception as e:
                    print(f"Error reading {fname}: {e}")

        if not documents:
            return jsonify({"summary": "❌ No valid documents found.", "similarity": 0, "image_url": None})

        ranked_indices, similarities = rank_documents_by_topic(documents, topic)
        top_idx = ranked_indices[0]
        summary = summarize_text(documents[top_idx])

        print(f"✔️ Summary for topic: {topic}\n{summary}")

        return jsonify({
            "summary": summary if summary.strip() else "⚠️ No summary extracted from top document.",
            "similarity": float(similarities[top_idx]),
            "image_url": None
        })


        return jsonify({
            "summary": summary,
            "similarity": similarities[top_idx],
            "image_url": None
        })


if __name__ == '__main__':
    app.run(debug=True)
