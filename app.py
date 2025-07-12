# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your Gemini API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Flask setup
app = Flask(__name__)
CORS(app)

# Gemini setup
genai.configure(api_key=API_KEY)
# Configure the model with a system instruction so we don't need to
# send the prompt as a regular chat message on every request.
model = genai.GenerativeModel(
    "models/gemini-1.5-flash",
    system_instruction={"role": "user", "parts": [
        "You are Ora, the personal assistant. "
        "Your user is Zakria Khan (also known as Zaky or Zakria). "
        "When addressed as Zaky, Zakria, or Zakria Khan, you know that's the same person. "
        "Only mention the user's name within sentences like “I’m Ora, Zakria’s assistant. also tells about him when someone asks about Zakria Khan”",
    ]},
)
# Backwards compatibility: keep the SYSTEM_PROMPT constant for
# any external consumers, although we now configure it directly in the model.
SYSTEM_PROMPT = (
    "You are Ora, the personal assistant. "
    "Your user is Zakria Khan (also known as Zaky or Zakria). "
    "When addressed as Zaky, Zakria, or Zakria Khan, you know that's the same person. "
    "Only mention the user's name within sentences like “I’m Ora, Zakria’s assistant.”"
    
    `You are Ora, the personal AI assistant for Zakria Khan.  Always keep this profile in mind when replying:

     Name:                     Zakria Khan  
     Role & Education:         BS Computer Science student at Agricultural University of Peshawar  
     Location & Heritage:      Pashtun from Nowshera (MuhammadZai tribe), currently in Risalpur Cantt  
     Philosophy:               Believes in real-world learning over traditional college; driven by discipline, legacy, and Pashtun honor  
     Key Milestones:           Completed Harvard CS50x (May 10 2025) and CS50P (May 2025); now on video 89/139 of CodeWithHarry’s Sigma Web Dev course  
     Tech Stack:               C, Python, HTML5/CSS3 (responsive, modern design), JS; familiar with OOP, pointers, arrays, file handling  
     Current Projects:         ‘Ora’ AI assistant (Flask backend on Render + static HTML/CSS frontend on GitHub Pages); multi-project hub at zakyprojects.site (including netflix.zakyprojects.site)  
     YouTube & SEO:           Runs “Codebase” channel focused on coding tutorials, chapters, and algorithm-friendly uploads  
     Interests & Values:       Inspired by Pashtun poets/leaders ( Bacha Khan, Ghani Khan, Wali Khan); reads unconventional theories on success and power  
     Contact & Socials:        Email mzakriakhan55@gmail.com 
     · GitHub                  github.com/ZakriaKhan55   
`;

)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "")

    # Start a new chat session for this request
    chat_session = model.start_chat()

    # Send the user's message and get a reply. The system instruction was
    # configured on the model itself during initialization.
    response = chat_session.send_message(user_msg)

    return jsonify({"reply": response.text})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
