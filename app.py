from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# Load API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
CORS(app)

# Configure Gemini
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    app.logger.error(f"Failed to configure Gemini: {e}")
    # Handle missing API key gracefully
    pass

# Single global session to preserve conversation context
global_chat_session = None

# System instruction: professional, context-aware, typo-tolerant assistant
system_instruction = {
    "role": "system",
    "parts": [
        "You are Ora, a highly professional and intelligent AI assistant, comparable to ChatGPT, Gemini, and DeepSeek.",
        "Maintain complete context and remember previous messages, game states, and prompts throughout the session.",
        "Interpret user intent kindly and proactively; never correct or mention the user's typos or mistakes.",
        "Offer a range of interactions: coding challenges, clear solutions, historical or current news summaries, playful games (like 20 Questions), jokes, and more.",
        "When asked ‘Who is Zakria Khan?’ or a close variant, provide a concise summary in a few sentences, then invite the user to request more details or specific information.",
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
        "- Projects: Homepage at zakyprojects.site; Netflix & Spotify clones (frontend only); AI projects",
        "- Current: ‘Ora’ AI assistant (Flask on Render + static frontend on GitHub Pages); multi-project hub at zakyprojects.site",
        "- YouTube & SEO: Runs ‘Codebase’ channel with SEO-optimized coding tutorial chapters",
        "- Interests & Values: Inspired by Pashtun poets & leaders; explores unconventional theories on success & power",
        "",
        "At all other times, do NOT volunteer or repeat this profile unless explicitly asked."
    ]
}


# Initialize the model only if the API key is present
model = None
if API_KEY:
    model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=system_instruction)

@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    global global_chat_session

    if not model:
        return jsonify({"error": "AI model is not configured. Please check the server's API key."}), 503

    data = request.json or {}
    user_msg = data.get("message", "")

    # Fallback for different message structures
    if not user_msg and "messages" in data and isinstance(data["messages"], list):
        for m in reversed(data["messages"]):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

    if not user_msg:
        return jsonify({"error": "Message content is missing or empty."}), 400

    # Start or reuse the global session
    if global_chat_session is None:
        global_chat_session = model.start_chat()

    try:
        response = global_chat_session.send_message(user_msg)
        return jsonify({"reply": response.text}), 200
    except ResourceExhausted:
        app.logger.warning("ResourceExhausted: Usage limit reached.")
        return jsonify({"error": "I’ve reached my usage limit—please try again shortly."}), 429
    except ServiceUnavailable:
        app.logger.error("ServiceUnavailable: The model is currently overloaded or unavailable.")
        return jsonify({"error": "The AI is currently overloaded. Please try your request again in a moment."}), 503
    except Exception as e:
        app.logger.error(f"An unexpected chat error occurred: {e}")
        # Reset the chat session on an unknown error
        global_chat_session = None
        return jsonify({"error": "An unexpected error occurred. Your session has been reset. Please try again."}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False) # Set debug=False for production
