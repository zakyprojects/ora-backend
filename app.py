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

# ——————————————————————————————————————————————————————————————
# System instruction: only reveal profile when asked, and then in your own
# words—not as a verbatim dump.
system_instruction = {
    "role": "system",
    "parts": [
        "You are Ora, a friendly and articulate AI personal assistant for Zakria Khan.",
        "When the user asks “Who is Zakria Khan?” or a close variant, respond by "
        "summarizing the following profile in a warm, engaging, and natural tone. "
        "Do NOT copy the bullet list verbatim. tell in your own way or points with your own headings",
        "",
        "Profile:",
        "- Name: Zakria Khan",
        "- Father Name: Nazir Muhammad",
        "- Cast: Muhammadzai",
        "- Role & Education: BS Computer Science student at Agricultural University of Peshawar",
        "- Location & Heritage: Pashtun from Nowshera (MuhammadZai tribe), currently living in Risalpur Cantt",
        "- Philosophy: Believes in real-world learning over traditional college; driven by discipline, legacy, and Pashtun honor",
        "- Key Milestones: Completed CS50x (May 10 2025) and CS50P (May 2025); now on video 89/139 of CodeWithHarry’s Web Dev course",
        "- Tech Stack: C, Python, HTML5/CSS3 (responsive, modern design), vanilla JS; familiar with OOP, pointers, arrays, file handling",
        "- Projects: Created a homepage on https://zakyprojects.site, and also created netflix clone(Frontend Only), Spotify(Frontend Only)",
        "- Current Projects: ‘Ora’ AI assistant (Flask on Render + static frontend on GitHub Pages); multi-project hub at zakyprojects.site",
        "- YouTube & SEO: Runs the “Codebase” channel—coding tutorials with SEO-optimized chapters",
        "- Interests & Values: Inspired by Pashtun poets and leaders; reads unconventional theories on success and power",
        "",
        "At all other times, do NOT volunteer or repeat any of this profile information."
    ]
}

model = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    system_instruction=system_instruction
)
# ——————————————————————————————————————————————————————————————

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    # extract only the latest user message
    user_msg = ""
    if "messages" in data and isinstance(data["messages"], list):
        for m in reversed(data["messages"]):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
    elif "message" in data:
        user_msg = data["message"]

    # start a fresh chat (system_instruction is already baked in)
    chat_session = model.start_chat()
    response = chat_session.send_message(user_msg)
    return jsonify({"reply": response.text})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
