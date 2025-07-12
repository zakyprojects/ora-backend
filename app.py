# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Tell Ora who “Zaky” is and how to use the name
SYSTEM_PROMPT = (
    "You are Ora, the personal assistant. "
    "Your user is Zakria Khan (also known as Zaky or Zakria). "
    "When addressed as Zaky, Zakria, or Zakria Khan, understand you refer to this user. "
    "Only mention the user's name in sentences like “I’m Ora, Zakria’s assistant.”"
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "")

    # Start chat with a context string (supported) rather than a system role
    chat_session = model.start_chat(context=SYSTEM_PROMPT)

    # Send the user's message and capture the response
    response = chat_session.send_message(user_msg)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
