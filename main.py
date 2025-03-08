from flask import Flask, request, jsonify
from src.model import Chatbot  # Import Chatbot from src

app = Flask(__name__)

# Initialize chatbot
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
