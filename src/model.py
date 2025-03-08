from rapidfuzz import process
import pickle
from utils import load_data
from sklearn.metrics.pairwise import cosine_similarity


dataset_path = r"D:\NewData\Projects\Hackathons\Reckon6\chatbot\dataset\emergency_data.json"
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

    def detect_intent(self, user_input):
        """Determines whether user input is an emergency, unclear, or non-emergency."""
        words = user_input.lower().split()

        # If input contains at least one known emergency word, consider it an emergency
        distress_keywords = ["help", "save", "urgent", "emergency", "stuck", "dying", "trapped", "can't breathe", "rescue"]
        if any(word in distress_keywords for word in words):
            return "emergency"

        # If input is too short, unclear, or gibberish, ask for clarification
        if len(words) < 3 or any(char.isdigit() for char in user_input):  # Filtering out junk
            return "unclear"

        return "non-emergency"

    def correct_and_match(self, user_input):
     """Finds the best matching keyword using fuzzy logic but only if it's a strong match."""
     match = process.extractOne(user_input, self.keyword_mapping.keys(), score_cutoff=85)

     if match:  # match is a tuple containing (best_match, score, *extra values*)
         best_match = match[0]  # Extract only the best match keyword
         return best_match
     return None


    def get_response(self, user_input):
        """Finds the best response based on emergency detection, similarity scoring, and intent confirmation."""
        user_input = user_input.lower().strip()

        # Step 1: Check if user is responding to a previous confirmation
        if self.last_question and user_input in ["yes", "yeah", "correct"]:
            self.last_question = None
            return self.last_intent  # Give the saved response

        if self.last_question and user_input in ["no", "nah", "wrong"]:
            self.last_question = None
            return "Sorry, please explain again. 🚨"

        # Step 2: Try fuzzy matching
        best_match = self.correct_and_match(user_input)
        if best_match:
            self.last_question = user_input
            self.last_intent = self.keyword_mapping[best_match]  # Save the expected response
            return f"Do you mean this: {best_match}? (Yes/No)"

        # Step 3: Fall back to TF-IDF similarity
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.X).flatten()

        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]

        print(f"🔍 Debug: TF-IDF Best Match Score = {best_match_score}")  # Debug print

        if best_match_score > 0.3:
            return self.answers[best_match_idx]

        return "I couldn't understand your request. Please describe the emergency more clearly, or call for help. 🚨"
