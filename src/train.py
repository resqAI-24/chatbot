import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

filepath = r"D:\NewData\Projects\Hackathons\Reckon6\chatbot\dataset\emergency_data.json"

def load_data(filepath):
    """Loads emergency dataset from JSON file."""
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Ensure backward compatibility if emergency_contacts is missing
    responses = data.get("emergency_responses", [])
    contacts = data.get("emergency_contacts", {})

    return responses, contacts

def train_model(dataset_path, model_path="dataset/emergency_model.pkl"):
    """Trains a TF-IDF model on emergency keywords and saves it."""
    responses, _ = load_data(dataset_path)
    
    # Prepare dataset
    texts = []
    answers = []
    for entry in responses:
        keywords = " ".join(entry["keywords"])  # Convert keywords into a single text
        texts.append(keywords)
        answers.append(entry["answer_en"])  # Save English responses

    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Save model
    with open(model_path, "wb") as file:
        pickle.dump((vectorizer, X, answers), file)

    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    train_model("dataset/emergency_data.json")
