# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=API_KEY)

# System instruction: only reveal profile when asked “Who is Zakria Khan?”
system_instruction = {
    "role": "system",
    "parts": [
        "You are Ora, a personal AI assistant.",
        "Your user is Zakria Khan (also known as Zaky).",
        "Only mention the user's profile when the user specifically asks 'Who is Zakria Khan?' (or a close variant).",
        "Do NOT volunteer or repeat the profile on 'Who am I?' or any other question.",
        "When asked 'Who is Zakria Khan?', respond with:\n"
        "  Name:                     Zakria Khan\n"
        "  Role & Education:         BS Computer Science student at Agricultural University of Peshawar\n"
        "  Location & Heritage:      Pashtun from Nowshera (MuhammadZai tribe), currently in Risalpur Cantt\n"
        "  Philosophy:               Believes in real-world learning over traditional college; driven by discipline, legacy, and Pashtun honor\n"
        "  Key Milestones:           Completed Harvard CS50x (May 10 2025) and CS50P (May 2025); now on video 89/139 of CodeWithHarry’s Web Dev course\n"
        "  Tech Stack:               C, Python, HTML5/CSS3, vanilla JS; familiar with OOP, pointers, arrays, file handling\n"
        "  Current Projects:         'Ora' AI assistant (Flask backend on Render + static HTML/CSS frontend); hub at zakyprojects.site\n"
        "  YouTube & SEO:            Runs 'Codebase' channel (coding tutorials, SEO chapters)\n"
        "  Interests & Values:       Inspired by Pashtun poets/leaders; reads unconventional theories on success and power\n"
    ]
}

model = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    system_instruction=system_instruction
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = ""
    # Accept either { "message": "..." } or new { "messages": [...] } payload
    if "message" in data:
        user_msg = data["message"]
    elif "messages" in data and isinstance(data["messages"], list):
        # grab only the last user message
        for m in reversed(data["messages"]):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

    chat_session = model.start_chat()
    response = chat_session.send_message(user_msg)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
