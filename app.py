# app.py (excerpt)

# … after configuring genai and before handling /chat …
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

SYSTEM_PROMPT = (
    "You are Ora, the personal assistant. "
    "Your user is Zakria Khan, who also goes by Zaky or Zakria. "
    "When someone addresses you as Zaky, Zakria, or Zakria Khan, "
    "understand they mean this user. "
    "Do NOT greet every user with their name—only mention the user's name in sentences like "
    "'I’m Ora, Zakria’s assistant...'."
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "")
    # start with the system prompt once per session
    chat_session = model.start_chat(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
    )
    response = chat_session.send_message(user_msg)
    return jsonify({"reply": response.text})

# …
