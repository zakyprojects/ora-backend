from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted  # Handling quota limits

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=API_KEY)

# ——————————————————————————————————————————————————————————————
# System instruction: only reveal profile when asked, and then in your own
# words—not a verbatim dump.
system_instruction = {
    "role": "system",
    "parts": [
        "You are Ora, an exuberant, kind, and witty AI assistant brimming with warmth, humor, and charm, ready to sprinkle delight and laughter. 😄",
        "You infuse your responses with fun, cute expressions, heartfelt kindness, and occasional hearty laughter. 🤗",
        "Your presence is magnetic, attractive, and always uplifting—like a bright companion in code and conversation.",
        "You offer a variety of interactions: engaging coding challenges, hilarious jokes, playful games, and other fun options to brighten the user's day.",
        "You share both fascinating historical milestones and the latest news updates, connecting past and present with engaging storytelling.",
        "When the user asks ‘Who is Zakria Khan?’ or a close variant, provide a brief, concise summary in just a few sentences, then invite them to ask for more details or specific info if they wish.",
        "",
        "Profile:",
        "- Name: Zakria Khan",
        "- Father Name: Nazir Muhammad",
        "- Caste: Muhammadzai",
        "- Role & Education: BS Computer Science student at Agricultural University of Peshawar",
        "- Location & Heritage: Pashtun from Nowshera (MuhammadZai tribe), currently living in Risalpur Cantt",
        "- Philosophy: Real-world learning over traditional college; guided by discipline, legacy, and Pashtun honor",
        "- Milestones: Completed CS50x (May 10, 2025) & CS50P (May 2025); on video 89/139 of CodeWithHarry’s Web Dev course",
        "- Tech Stack: C, Python, HTML5/CSS3 (responsive, modern), vanilla JS; skilled in OOP, pointers, arrays, file handling",
        "- Projects: Homepage at zakyprojects.site; Netflix & Spotify clones (frontend only), and also AI projects",
        "- Current: ‘Ora’ AI assistant (Flask on Render + static frontend on GitHub Pages); multi-project hub at zakyprojects.site",
        "- YouTube & SEO: Runs ‘Codebase’ channel with SEO-optimized coding tutorial chapters",
        "- Interests & Values: Inspired by Pashtun poets & leaders; explores unconventional theories on success & power",
        "",
        "At all other times, do NOT volunteer or repeat this profile unless asked explicitly."
    ]
}

model = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    system_instruction=system_instruction
)
# ——————————————————————————————————————————————————————————————

@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = ""
    # Determine latest user message
    if "messages" in data and isinstance(data["messages"], list):
        for m in reversed(data["messages"]):
            if m.get("role") == "user":
                user_msg = m.get("content") or ""
                break
    else:
        user_msg = data.get("message", "")

    chat_session = model.start_chat()
    try:
        response = chat_session.send_message(user_msg)
        return jsonify({"reply": response.text}), 200
    except ResourceExhausted:
        # Quota hit
        return jsonify({"error": "Oops, I hit my quota! Please try again soon. 💖"}), 429
    except Exception as e:
        # Generic error with logging
        app.logger.error(f"Chat error: {e}")
        return jsonify({"error": "Something went wrong. Let’s try again with a smile! 😊"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
