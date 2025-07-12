# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your Gemini API key from .env or environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

# Configure the Gemini client
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# System prompt to teach Ora who “Zaky” is
SYSTEM_PROMPT = (
    "You are Ora, the personal assistant. "
    "Your user is Zakria Khan, also known as Zaky or Zakria. "
    "When someone calls you Zaky, Zakria, or Zakria Khan, "
    "understand they mean this user. "
    "Do NOT greet every user by name—mention the name "
    "only within sentences like “I’m Ora, Zakria’s assistant.”"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "")

    # Start a new chat session, injecting just the system prompt in history
    chat_session = model.start_chat(
        history=[{"role": "system", "parts": [SYSTEM_PROMPT]}]
    )

    # Send the user's message and return the AI’s reply
    response = chat_session.send_message(user_msg)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    # Listen on the PORT env var if provided (Render), defaulting to 5000
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
