import os
from flask import Flask, render_template, request, jsonify
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    # Create LiveAvatar embed session
    resp = requests.post(
        "https://api.liveavatar.com/v2/embeddings",
        headers={
            "X-API-KEY": LIVEAVATAR_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "avatar_id": LIVEAVATAR_AVATAR_ID,
            "is_sandbox": True
        }
    )

    if resp.status_code != 200:
        return jsonify({"error": resp.text}), 500

    result = resp.json()
    embed_url = result["data"]["url"]
    return jsonify({"embed_url": embed_url})

@app.route("/el/signed_url", methods=["GET"])
def el_signed_url():
    resp = requests.get(
        "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        params={"agent_id": ELEVENLABS_AGENT_ID}
    )
    if resp.status_code != 200:
        return jsonify({"error": resp.text}), 500
    return jsonify(resp.json())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
