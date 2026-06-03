import os
from flask import Flask, render_template, request, jsonify
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID")
LIVEAVATAR_CONTEXT_ID = os.getenv("LIVEAVATAR_CONTEXT_ID")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    data = request.json or {}
    user_name = data.get("user_name", "Guest")

    # Create LiveAvatar embed session
    resp = requests.post(
        "https://api.liveavatar.com/v2/embeddings",
        headers={
            "X-API-KEY": LIVEAVATAR_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "avatar_id": LIVEAVATAR_AVATAR_ID,
            "context_id": LIVEAVATAR_CONTEXT_ID,
            "is_sandbox": True
        }
    )

    if resp.status_code != 200:
        return jsonify({"error": resp.text}), 500

    result = resp.json()
    embed_url = result["data"]["url"]
    return jsonify({"embed_url": embed_url, "user_name": user_name})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
