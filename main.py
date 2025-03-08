from flask import Flask, request, jsonify
from src.model import Chatbot  # Import Chatbot from src
import json
import pickle
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
from utils import load_data

app = Flask(__name__)
# Initialize chatbot
bot = Chatbot("dataset/emergency_data.json")
class Chatbot:
    def __init__(self, dataset_path, model_path="dataset/emergency_model.pkl"):
        """Initialize chatbot with emergency dataset."""
        self.responses, self.contacts = load_data(dataset_path)

        # Ensure self.contacts exists even if missing from dataset
        if not self.contacts:
            self.contacts = {"message": "Emergency contacts are not available in the dataset."}

        # Load trained TF-IDF model
        with open(model_path, "rb") as file:
            self.vectorizer, self.X, self.answers = pickle.load(file)

        # Collect all possible keywords for fuzzy matching
        self.keyword_mapping = {}
        for entry in self.responses:
            for keyword in entry["keywords"]:
                self.keyword_mapping[keyword] = entry["answer_en"]

        # Stores last question & suggested intent
        self.last_question = None
        self.last_intent = None

    def correct_and_match(self, user_input):
        """Finds the best matching keyword using fuzzy logic but only if it's a strong match."""
        match = process.extractOne(user_input, self.keyword_mapping.keys(), score_cutoff=70)
        if match:
            return match[0]
        return None

    def get_response(self, user_input):
        """Finds the best response based on emergency detection, similarity scoring, and intent confirmation."""
        user_input = user_input.lower().strip()

        # Try fuzzy matching first
        best_match = self.correct_and_match(user_input)
        if best_match:
            return self.keyword_mapping[best_match]

        # Fall back to TF-IDF similarity model
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.X).flatten()

        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]

        if best_match_score > 0.3:
            return self.answers[best_match_idx]

        return "I couldn't understand your request. Please describe the emergency more clearly, or call for help. 🚨"

bot = Chatbot("dataset/emergency_data.json")

@app.route("/chat", methods=["POST"])
def chat():
    """Receives user input and returns chatbot response."""
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"response": "Please provide a message."}), 400

    response = bot.get_response(user_input)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "🔴 ResQAI Emergency Chatbot is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
