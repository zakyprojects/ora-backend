# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from google.generativeai import Part
from google.generativeai import Content  # import the Content class for chat history
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Flask setup
app = Flask(__name__)
CORS(app)

# Gemini setup
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

SYSTEM_PROMPT = (
    "You are Ora, the personal assistant. "
    "Your user is Zakria Khan (also known as Zaky or Zakria). "
    "When addressed as Zaky, Zakria, or Zakria Khan, you know that's the same person. "
    "Only mention the user's name within sentences like “I’m Ora, Zakria’s assistant.”"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "")

    # Build a one-item history containing our system instruction
    init = Content(
        role="system",
        parts=[Part.from_text(SYSTEM_PROMPT)]
    )
    # Start chat with that history
    chat_session = model.start_chat(history=[init])

    # Now send the actual user message
    response = chat_session.send_message(user_msg)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
