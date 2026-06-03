import os
import sqlite3
import uuid
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response, g
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID")
LIVEAVATAR_CONTEXT_ID = os.getenv("LIVEAVATAR_CONTEXT_ID")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

DATABASE = os.path.join(os.path.dirname(__file__), "data.db")

# ── Face recognition setup ───────────────────────────────────────────────────
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("face_recognition not installed — face ID disabled")

# ── Database ─────────────────────────────────────────────────────────────────

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with sqlite3.connect(DATABASE) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT DEFAULT 'Guest',
                face_encoding TEXT,
                first_seen TEXT,
                last_seen TEXT,
                visit_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                avatar_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                speaker TEXT,
                text TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)

init_db()

# ── Face recognition helpers ─────────────────────────────────────────────────

def decode_image(b64_string):
    """Decode base64 image to numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    import cv2
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_face_encoding(img_rgb):
    """Get 128-d face encoding from image."""
    encodings = face_recognition.face_encodings(img_rgb)
    if not encodings:
        return None
    return encodings[0]

def find_matching_user(db, encoding, tolerance=0.5):
    """Compare encoding against all stored users."""
    users = db.execute("SELECT id, name, face_encoding, visit_count FROM users WHERE face_encoding IS NOT NULL").fetchall()
    for user in users:
        stored = np.array(eval(user["face_encoding"]))
        dist = face_recognition.face_distance([stored], encoding)[0]
        if dist < tolerance:
            return dict(user)
    return None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/identify", methods=["POST"])
def identify():
    """
    Receive base64 camera snapshot.
    Match against stored faces.
    If match found: return user info + history.
    If new face: create new user, store encoding.
    """
    data = request.json or {}
    image_b64 = data.get("image")
    name = data.get("name", "Guest")

    if not image_b64 or not FACE_RECOGNITION_AVAILABLE:
        user_id = str(uuid.uuid4())
        return jsonify({"user_id": user_id, "name": "Guest", "returning": False})

    try:
        img_rgb = decode_image(image_b64)
        encoding = get_face_encoding(img_rgb)
    except Exception as e:
        user_id = str(uuid.uuid4())
        return jsonify({"user_id": user_id, "name": "Guest", "returning": False, "error": str(e)})

    db = get_db()

    if encoding is not None:
        match = find_matching_user(db, encoding)
        if match:
            # Returning user
            db.execute("UPDATE users SET last_seen = ?, visit_count = visit_count + 1 WHERE id = ?",
                       (datetime.utcnow().isoformat(), match["id"]))
            db.commit()
            return jsonify({
                "user_id": match["id"],
                "name": match["name"],
                "returning": True,
                "visit_count": match["visit_count"] + 1
            })
        else:
            # New user — store face
            user_id = str(uuid.uuid4())
            encoding_str = str(encoding.tolist())
            db.execute(
                "INSERT INTO users (id, name, face_encoding, first_seen, last_seen, visit_count) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, encoding_str, datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), 1)
            )
            db.commit()
            return jsonify({"user_id": user_id, "name": name, "returning": False})
    else:
        # No face detected
        user_id = str(uuid.uuid4())
        return jsonify({"user_id": user_id, "name": "Guest", "returning": False, "face_detected": False})

@app.route("/start_session", methods=["POST"])
def start_session():
    data = request.json or {}
    user_id = data.get("user_id") or str(uuid.uuid4())

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        db.execute(
            "INSERT INTO users (id, name, first_seen, last_seen) VALUES (?, ?, ?, ?)",
            (user_id, "Guest", datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
        )
        db.commit()
    else:
        db.execute("UPDATE users SET last_seen = ? WHERE id = ?",
                   (datetime.utcnow().isoformat(), user_id))
        db.commit()

    session_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO sessions (id, user_id, avatar_id, started_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, LIVEAVATAR_AVATAR_ID, datetime.utcnow().isoformat())
    )
    db.commit()

    history = db.execute(
        "SELECT speaker, text FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 20",
        (user_id,)
    ).fetchall()
    history = [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(history)]

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

    return jsonify({
        "embed_url": embed_url,
        "session_id": session_id,
        "user_id": user_id,
        "history": history,
        "returning": user is not None
    })

@app.route("/save_message", methods=["POST"])
def save_message():
    data = request.json or {}
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    speaker = data.get("speaker")
    text = data.get("text")

    if not all([session_id, user_id, speaker, text]):
        return jsonify({"error": "Missing fields"}), 400

    db = get_db()
    db.execute(
        "INSERT INTO messages (session_id, user_id, speaker, text, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, user_id, speaker, text, datetime.utcnow().isoformat())
    )
    db.commit()
    return jsonify({"ok": True})

@app.route("/history/<user_id>")
def get_history(user_id):
    db = get_db()
    messages = db.execute(
        "SELECT speaker, text, timestamp FROM messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
        (user_id,)
    ).fetchall()
    return jsonify([dict(m) for m in messages])

@app.route("/el/signed_url")
def el_signed_url():
    resp = requests.get(
        "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        params={"agent_id": ELEVENLABS_AGENT_ID}
    )
    if resp.status_code != 200:
        return jsonify({"error": resp.text}), 500
    return jsonify(resp.json())

@app.route("/avatar/thumbnail")
def avatar_thumbnail():
    try:
        resp = requests.get(
            f"https://api.liveavatar.com/v2/avatars/{LIVEAVATAR_AVATAR_ID}",
            headers={"X-API-KEY": LIVEAVATAR_API_KEY}
        )
        if resp.status_code == 200:
            data = resp.json()
            thumb = data.get("data", {}).get("thumbnail_url") or data.get("data", {}).get("preview_url")
            if thumb:
                img = requests.get(thumb)
                return Response(img.content, mimetype=img.headers.get("Content-Type", "image/jpeg"))
    except Exception:
        pass
    local = os.path.join(os.path.dirname(__file__), "static", "petar.jpg")
    if os.path.exists(local):
        return send_file(local, mimetype="image/jpeg")
    return jsonify({"error": "No image"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
