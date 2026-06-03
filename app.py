<<<<<<< HEAD
import os
import sqlite3
import uuid
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response, g
=======
import os, sqlite3, uuid, base64, hashlib, json, threading, time
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, Response, g, session, redirect, url_for
from flask_cors import CORS
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
<<<<<<< HEAD

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
=======
app.secret_key = os.getenv("SECRET_KEY", "trione-secret-change-in-prod")
CORS(app)

DATABASE = os.path.join(os.path.dirname(__file__), "data.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LIVEAVATAR_API_KEY = os.getenv("LIVEAVATAR_API_KEY")

# ── DB ────────────────────────────────────────────────────────────────────────
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
<<<<<<< HEAD
    if db:
        db.close()
=======
    if db: db.close()
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)

def init_db():
    with sqlite3.connect(DATABASE) as db:
        db.executescript("""
<<<<<<< HEAD
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT DEFAULT 'Guest',
                face_encoding TEXT,
=======
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                avatar_id TEXT,
                context_id TEXT,
                elevenlabs_agent_id TEXT,
                persona_summary TEXT,
                ego_model TEXT,
                created_at TEXT,
                active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                face_encoding TEXT,
                name TEXT DEFAULT 'Guest',
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)
                first_seen TEXT,
                last_seen TEXT,
                visit_count INTEGER DEFAULT 1
            );
<<<<<<< HEAD

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
=======
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                user_id TEXT,
                session_type TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                user_id TEXT,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                chunk TEXT NOT NULL,
                source_session_id TEXT,
                category TEXT,
                ego_layer TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS ego_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT NOT NULL,
                revision TEXT NOT NULL,
                trigger_text TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS training_videos (
                id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                file_path TEXT,
                transcript TEXT,
                created_at TEXT,
                delete_after TEXT,
                deleted INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS admins (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)
            );
        """)

init_db()

<<<<<<< HEAD
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
=======
# ── Seed default admin ────────────────────────────────────────────────────────

def seed_admin():
    with sqlite3.connect(DATABASE) as db:
        count = db.execute("SELECT COUNT(*) FROM admins").fetchone()[0]
        if count == 0:
            db.execute(
                "INSERT INTO admins (id, email, password_hash, created_at) VALUES (?,?,?,?)",
                (str(uuid.uuid4()), "admin@example.com",
                 hashlib.sha256("password".encode()).hexdigest(),
                 datetime.utcnow().isoformat())
            )
            db.commit()
            print("Default admin created: admin@example.com / password")

seed_admin()

# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_customer(db, cid):
    return db.execute("SELECT * FROM customers WHERE id=?", (cid,)).fetchone()

def get_first_customer(db):
    return db.execute("SELECT * FROM customers WHERE active=1 LIMIT 1").fetchone()

def liveavatar_embed(avatar_id, context_id, sandbox=True):
    if not avatar_id or not context_id:
        return None, "Missing avatar_id or context_id"
    r = requests.post(
        "https://api.liveavatar.com/v2/embeddings",
        headers={"X-API-KEY": LIVEAVATAR_API_KEY, "Content-Type": "application/json"},
        json={"avatar_id": avatar_id, "context_id": context_id, "is_sandbox": sandbox}
    )
    if r.status_code != 200:
        return None, r.text
    return r.json()["data"]["url"], None

def gemini(prompt, system=None):
    if not GEMINI_API_KEY:
        return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    r = requests.post(url, json={"contents": contents})
    if r.status_code == 200:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    return ""

def parse_json_safe(text):
    text = text.strip().strip("```json").strip("```").strip()
    try:
        return json.loads(text)
    except:
        return None

# ── EGO Engine ────────────────────────────────────────────────────────────────
# The core of the system. Every time Petar speaks, we don't just store what he said.
# We reason about what it reveals about who he is — his beliefs, contradictions,
# relationship dynamics, reasoning patterns and values.
# Over time this builds a living model of his EGO that the avatar embodies.

def extract_ego(text, existing_ego=None, speaker_context="general conversation"):
    """
    Extract structured EGO data from Petar's speech.
    Returns dict with beliefs, contradictions, relationships, reasoning_patterns, values, raw_chunks.
    """
    ego_summary = existing_ego or "No existing EGO model yet."

    prompt = f"""You are building an Artificial EGO — a living model of a person's identity, reasoning and relationships.

Existing EGO model:
{ego_summary}

New input from the person:
"{text}"

Context: {speaker_context}

Analyze this input deeply. Extract:
1. beliefs: Core beliefs, values or worldviews revealed or reinforced
2. contradictions: Anything that contradicts or evolves from existing beliefs (growth, nuance, inconsistency)
3. relationships: How they speak/think about specific people or groups — dynamics, loyalty, tension
4. reasoning_patterns: How they think through problems — deductive, intuitive, pragmatic, philosophical
5. values: What they prioritize — family, money, truth, power, connection, legacy, freedom
6. raw_chunks: Simple factual statements to store as knowledge (beliefs, experiences, preferences, stories)

Return ONLY valid JSON, no markdown:
{{
  "beliefs": ["..."],
  "contradictions": ["..."],
  "relationships": {{"person_or_group": "dynamic description"}},
  "reasoning_patterns": ["..."],
  "values": ["..."],
  "raw_chunks": [{{"chunk": "...", "category": "belief|experience|preference|fact|story|relationship"}}]
}}"""

    raw = gemini(prompt)
    result = parse_json_safe(raw)
    if not result:
        # Fallback — simple extraction
        return {
            "beliefs": [], "contradictions": [], "relationships": {},
            "reasoning_patterns": [], "values": [],
            "raw_chunks": [{"chunk": text[:500], "category": "fact"}]
        }
    return result


def update_ego_model(db, customer_id, new_ego_data, trigger_text):
    """
    Merge new EGO data into the existing EGO model.
    The EGO model is a living JSON document that grows and self-corrects.
    """
    customer = get_customer(db, customer_id)
    existing_ego_raw = customer["ego_model"] if customer and customer["ego_model"] else None

    if existing_ego_raw:
        try:
            existing_ego = json.loads(existing_ego_raw)
        except:
            existing_ego = {}
    else:
        existing_ego = {
            "beliefs": [], "contradictions": [], "relationships": {},
            "reasoning_patterns": [], "values": [], "summary": ""
        }

    # Merge arrays — deduplicate
    for key in ["beliefs", "reasoning_patterns", "values"]:
        existing = existing_ego.get(key, [])
        new = new_ego_data.get(key, [])
        merged = list({item for item in existing + new})
        existing_ego[key] = merged

    # Merge relationships
    existing_rels = existing_ego.get("relationships", {})
    new_rels = new_ego_data.get("relationships", {})
    for person, dynamic in new_rels.items():
        if person in existing_rels:
            existing_rels[person] = existing_rels[person] + " | " + dynamic
        else:
            existing_rels[person] = dynamic
    existing_ego["relationships"] = existing_rels

    # Track contradictions as growth
    contradictions = existing_ego.get("contradictions", [])
    contradictions.extend(new_ego_data.get("contradictions", []))
    existing_ego["contradictions"] = contradictions[-20:]  # keep last 20

    # Regenerate summary from full EGO
    summary_prompt = f"""Based on this EGO model, write a 3-sentence summary of this person's core identity, 
how they think, and what drives them. Be specific and insightful, not generic.

EGO model: {json.dumps(existing_ego, indent=2)}"""
    summary = gemini(summary_prompt)
    existing_ego["summary"] = summary
    existing_ego["last_updated"] = datetime.utcnow().isoformat()

    ego_json = json.dumps(existing_ego)
    db.execute("UPDATE customers SET ego_model=?, persona_summary=? WHERE id=?",
               (ego_json, summary, customer_id))

    # Store revision history
    db.execute(
        "INSERT INTO ego_revisions (customer_id, revision, trigger_text, created_at) VALUES (?,?,?,?)",
        (customer_id, ego_json, trigger_text[:500], datetime.utcnow().isoformat())
    )
    db.commit()
    return existing_ego


def get_ego_context(db, customer_id, max_chunks=40):
    """
    Build the full context for the avatar — EGO model + recent knowledge.
    This is what gets passed to the avatar so it can reason as Petar would.
    """
    customer = get_customer(db, customer_id)
    ego_model = None
    if customer and customer["ego_model"]:
        try:
            ego_model = json.loads(customer["ego_model"])
        except:
            pass

    chunks = db.execute(
        "SELECT chunk, category, ego_layer FROM knowledge_base WHERE customer_id=? ORDER BY created_at DESC LIMIT ?",
        (customer_id, max_chunks)
    ).fetchall()

    context_parts = []
    if ego_model:
        context_parts.append(f"=== PERSONA ===\n{ego_model.get('summary', '')}")
        if ego_model.get("beliefs"):
            context_parts.append("=== CORE BELIEFS ===\n" + "\n".join(f"• {b}" for b in ego_model["beliefs"][:10]))
        if ego_model.get("values"):
            context_parts.append("=== VALUES ===\n" + "\n".join(f"• {v}" for v in ego_model["values"][:8]))
        if ego_model.get("reasoning_patterns"):
            context_parts.append("=== HOW THEY THINK ===\n" + "\n".join(f"• {r}" for r in ego_model["reasoning_patterns"][:6]))
        if ego_model.get("relationships"):
            rels = ego_model["relationships"]
            rel_text = "\n".join(f"• {p}: {d}" for p, d in list(rels.items())[:10])
            context_parts.append(f"=== RELATIONSHIPS ===\n{rel_text}")

    if chunks:
        chunk_text = "\n".join([f"[{c['category']}] {c['chunk']}" for c in chunks])
        context_parts.append(f"=== KNOWLEDGE ===\n{chunk_text}")

    return "\n\n".join(context_parts)


# ── Video cleanup ─────────────────────────────────────────────────────────────

def cleanup_videos():
    while True:
        try:
            with sqlite3.connect(DATABASE) as db:
                now = datetime.utcnow().isoformat()
                old = db.execute(
                    "SELECT * FROM training_videos WHERE delete_after < ? AND deleted=0", (now,)
                ).fetchall()
                for v in old:
                    if v[3] and os.path.exists(v[3]):
                        os.remove(v[3])
                    db.execute("UPDATE training_videos SET deleted=1, file_path=NULL WHERE id=?", (v[0],))
                db.commit()
        except Exception as e:
            print(f"Cleanup error: {e}")
        time.sleep(3600)

threading.Thread(target=cleanup_videos, daemon=True).start()

# ── Face recognition ──────────────────────────────────────────────────────────

def encode_face(b64):
    try:
        import cv2, face_recognition
        if "," in b64: b64 = b64.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(img_rgb)
        return encs[0] if encs else None
    except Exception as e:
        print(f"Face encode error: {e}")
        return None

def find_face(db, enc, tolerance=0.5):
    try:
        import face_recognition
        users = db.execute("SELECT * FROM users WHERE face_encoding IS NOT NULL").fetchall()
        for u in users:
            stored = np.array(json.loads(u["face_encoding"]))
            if face_recognition.face_distance([stored], enc)[0] < tolerance:
                return dict(u)
    except Exception as e:
        print(f"Face match error: {e}")
    return None

# ════════════════════════════════════════════════════════════════════════════
# USER PORTAL
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def user_home():
    return render_template("user/index.html")

@app.route("/u/identify", methods=["POST"])
def user_identify():
    data = request.json or {}
    b64 = data.get("image")
    db = get_db()
    user_id = str(uuid.uuid4())
    returning = False
    name = "Guest"

    if b64:
        enc = encode_face(b64)
        if enc is not None:
            match = find_face(db, enc)
            if match:
                user_id = match["id"]
                name = match["name"]
                returning = True
                db.execute("UPDATE users SET last_seen=?, visit_count=visit_count+1 WHERE id=?",
                           (datetime.utcnow().isoformat(), user_id))
            else:
                db.execute(
                    "INSERT INTO users (id, face_encoding, name, first_seen, last_seen) VALUES (?,?,?,?,?)",
                    (user_id, json.dumps(enc.tolist()), "Guest",
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
                )
            db.commit()

    history = db.execute(
        "SELECT speaker, text FROM messages WHERE user_id=? ORDER BY timestamp DESC LIMIT 20",
        (user_id,)
    ).fetchall()

    return jsonify({
        "user_id": user_id,
        "name": name,
        "returning": returning,
        "history": [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(history)]
    })

@app.route("/u/session", methods=["POST"])
def user_session():
    data = request.json or {}
    user_id = data.get("user_id", str(uuid.uuid4()))
    db = get_db()

    customer = get_first_customer(db)
    if not customer:
        return jsonify({"error": "No active avatar found"}), 404

    rag_context = get_ego_context(db, customer["id"])

    session_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO sessions (id, customer_id, user_id, session_type, started_at) VALUES (?,?,?,?,?)",
        (session_id, customer["id"], user_id, "user_chat", datetime.utcnow().isoformat())
    )
    db.commit()

    embed_url, err = liveavatar_embed(customer["avatar_id"], customer["context_id"])
    if err:
        return jsonify({"error": err}), 500
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)

    return jsonify({
        "embed_url": embed_url,
        "session_id": session_id,
<<<<<<< HEAD
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
=======
        "customer_id": customer["id"],
        "customer_name": customer["name"],
        "rag_context": rag_context,
        "persona": customer["persona_summary"] or ""
    })

@app.route("/u/message", methods=["POST"])
def user_message():
    data = request.json or {}
    db = get_db()
    db.execute(
        "INSERT INTO messages (session_id, customer_id, user_id, speaker, text, timestamp) VALUES (?,?,?,?,?,?)",
        (data.get("session_id"), data.get("customer_id"), data.get("user_id"),
         data.get("speaker"), data.get("text"), datetime.utcnow().isoformat())
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)
    )
    db.commit()
    return jsonify({"ok": True})

<<<<<<< HEAD
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
=======
# ════════════════════════════════════════════════════════════════════════════
# CUSTOMER PORTAL
# ════════════════════════════════════════════════════════════════════════════

@app.route("/customer/login", methods=["GET", "POST"])
def customer_login():
    if request.method == "POST":
        data = request.json or {}
        db = get_db()
        c = db.execute(
            "SELECT * FROM customers WHERE email=? AND password_hash=?",
            (data.get("email"), hash_pw(data.get("password", "")))
        ).fetchone()
        if c:
            session["customer_id"] = c["id"]
            return jsonify({"ok": True, "name": c["name"]})
        return jsonify({"error": "Invalid credentials"}), 401
    return render_template("customer/login.html")

@app.route("/customer/logout")
def customer_logout():
    session.pop("customer_id", None)
    return redirect(url_for("customer_login"))

@app.route("/customer/")
def customer_home():
    if "customer_id" not in session:
        return redirect(url_for("customer_login"))
    return render_template("customer/index.html")

@app.route("/customer/session", methods=["POST"])
def customer_session():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    customer = get_customer(db, session["customer_id"])
    if not customer:
        return jsonify({"error": "Not found"}), 404

    session_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO sessions (id, customer_id, session_type, started_at) VALUES (?,?,?,?)",
        (session_id, customer["id"], "customer_training", datetime.utcnow().isoformat())
    )
    db.commit()

    embed_url, err = liveavatar_embed(customer["avatar_id"], customer["context_id"])
    if err:
        return jsonify({"error": err}), 500

    knowledge_count = db.execute(
        "SELECT COUNT(*) as cnt FROM knowledge_base WHERE customer_id=?", (customer["id"],)
    ).fetchone()["cnt"]

    # Get current EGO model
    ego_model = None
    if customer["ego_model"]:
        try:
            ego_model = json.loads(customer["ego_model"])
        except:
            pass

    return jsonify({
        "embed_url": embed_url,
        "session_id": session_id,
        "customer_id": customer["id"],
        "knowledge_count": knowledge_count,
        "persona": customer["persona_summary"] or "",
        "ego_model": ego_model
    })

@app.route("/customer/train", methods=["POST"])
def customer_train():
    """
    Core EGO training endpoint.
    Called every time Petar speaks — either via voice transcript or manual text.
    Extracts structured EGO data and updates the living EGO model.
    """
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json or {}
    text = data.get("text", "").strip()
    session_id = data.get("session_id")
    speaker_context = data.get("context", "general conversation")

    if not text or len(text) < 5:
        return jsonify({"ok": True, "chunks_added": 0})

    db = get_db()
    customer_id = session["customer_id"]

    # Save raw message
    db.execute(
        "INSERT INTO messages (session_id, customer_id, speaker, text, timestamp) VALUES (?,?,?,?,?)",
        (session_id, customer_id, "customer", text, datetime.utcnow().isoformat())
    )
    db.commit()

    # Get existing EGO for context
    customer = get_customer(db, customer_id)
    existing_ego_summary = customer["persona_summary"] or "No existing model yet."

    # Extract structured EGO data
    ego_data = extract_ego(text, existing_ego_summary, speaker_context)

    # Store raw knowledge chunks
    chunks_added = 0
    for chunk in ego_data.get("raw_chunks", []):
        if chunk.get("chunk"):
            db.execute(
                "INSERT INTO knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (?,?,?,?,?,?)",
                (customer_id, chunk["chunk"], session_id,
                 chunk.get("category", "fact"), "raw", datetime.utcnow().isoformat())
            )
            chunks_added += 1

    # Store EGO-layer items
    for belief in ego_data.get("beliefs", []):
        db.execute(
            "INSERT INTO knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (?,?,?,?,?,?)",
            (customer_id, belief, session_id, "belief", "ego", datetime.utcnow().isoformat())
        )
    for value in ego_data.get("values", []):
        db.execute(
            "INSERT INTO knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (?,?,?,?,?,?)",
            (customer_id, value, session_id, "value", "ego", datetime.utcnow().isoformat())
        )
    db.commit()

    # Update living EGO model in background thread
    def bg_update():
        with sqlite3.connect(DATABASE) as bg_db:
            bg_db.row_factory = sqlite3.Row
            update_ego_model(bg_db, customer_id, ego_data, text)

    threading.Thread(target=bg_update, daemon=True).start()

    return jsonify({
        "ok": True,
        "chunks_added": chunks_added,
        "beliefs_extracted": len(ego_data.get("beliefs", [])),
        "values_extracted": len(ego_data.get("values", [])),
        "contradictions": len(ego_data.get("contradictions", []))
    })

@app.route("/customer/ego")
def customer_ego():
    """Return the full EGO model for the customer."""
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    customer = get_customer(db, session["customer_id"])
    if not customer or not customer["ego_model"]:
        return jsonify({"ego": None, "summary": ""})
    try:
        ego = json.loads(customer["ego_model"])
    except:
        ego = {}
    return jsonify({"ego": ego, "summary": customer["persona_summary"] or ""})

@app.route("/customer/ego/revisions")
def ego_revisions():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    revisions = db.execute(
        "SELECT id, trigger_text, created_at FROM ego_revisions WHERE customer_id=? ORDER BY created_at DESC LIMIT 20",
        (session["customer_id"],)
    ).fetchall()
    return jsonify([dict(r) for r in revisions])

@app.route("/customer/knowledge")
def customer_knowledge():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    chunks = db.execute(
        "SELECT id, chunk, category, ego_layer, created_at FROM knowledge_base WHERE customer_id=? ORDER BY created_at DESC",
        (session["customer_id"],)
    ).fetchall()
    return jsonify([dict(c) for c in chunks])

@app.route("/customer/knowledge/<int:kid>", methods=["DELETE"])
def delete_knowledge(kid):
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    db.execute("DELETE FROM knowledge_base WHERE id=? AND customer_id=?", (kid, session["customer_id"]))
    db.commit()
    return jsonify({"ok": True})

@app.route("/customer/sessions")
def customer_sessions():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    rows = db.execute(
        "SELECT s.id, s.session_type, s.started_at, COUNT(m.id) as message_count "
        "FROM sessions s LEFT JOIN messages m ON s.id=m.session_id "
        "WHERE s.customer_id=? GROUP BY s.id ORDER BY s.started_at DESC LIMIT 50",
        (session["customer_id"],)
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/customer/me")
def customer_me():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    c = get_customer(db, session["customer_id"])
    ego = None
    if c["ego_model"]:
        try: ego = json.loads(c["ego_model"])
        except: pass
    return jsonify({
        "name": c["name"],
        "email": c["email"],
        "persona": c["persona_summary"] or "",
        "ego": ego
    })

# ════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ════════════════════════════════════════════════════════════════════════════

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        data = request.json or {}
        db = get_db()
        a = db.execute(
            "SELECT * FROM admins WHERE email=? AND password_hash=?",
            (data.get("email"), hash_pw(data.get("password", "")))
        ).fetchone()
        if a:
            session["admin_id"] = a["id"]
            return jsonify({"ok": True})
        return jsonify({"error": "Invalid credentials"}), 401
    return render_template("admin/login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_id", None)
    return redirect(url_for("admin_login"))

@app.route("/admin/")
def admin_home():
    if "admin_id" not in session:
        return redirect(url_for("admin_login"))
    return render_template("admin/index.html")

@app.route("/admin/customers", methods=["GET"])
def admin_customers():
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    customers = db.execute(
        "SELECT id, name, email, active, created_at FROM customers ORDER BY created_at DESC"
    ).fetchall()
    return jsonify([dict(c) for c in customers])

@app.route("/admin/customers", methods=["POST"])
def admin_create_customer():
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.json or {}
    db = get_db()
    cid = str(uuid.uuid4())
    db.execute(
        "INSERT INTO customers (id, name, email, password_hash, avatar_id, context_id, elevenlabs_agent_id, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (cid, data["name"], data["email"], hash_pw(data["password"]),
         data.get("avatar_id"), data.get("context_id"),
         data.get("elevenlabs_agent_id"), datetime.utcnow().isoformat())
    )
    db.commit()
    return jsonify({"ok": True, "id": cid})

@app.route("/admin/toggle_customer/<cid>", methods=["POST"])
def admin_toggle_customer(cid):
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    db.execute("UPDATE customers SET active=1-active WHERE id=?", (cid,))
    db.commit()
    return jsonify({"ok": True})

@app.route("/admin/stats")
def admin_stats():
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    return jsonify({
        "total_customers": db.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
        "total_users": db.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "total_sessions": db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        "total_messages": db.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "total_knowledge_chunks": db.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0],
        "total_ego_revisions": db.execute("SELECT COUNT(*) FROM ego_revisions").fetchone()[0],
    })

@app.route("/admin/customer/<cid>/ego")
def admin_customer_ego(cid):
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    c = get_customer(db, cid)
    if not c:
        return jsonify({"error": "Not found"}), 404
    ego = None
    if c["ego_model"]:
        try: ego = json.loads(c["ego_model"])
        except: pass
    return jsonify({"name": c["name"], "persona": c["persona_summary"], "ego": ego})

# ── Shared ────────────────────────────────────────────────────────────────────

@app.route("/el/signed_url")
def el_signed_url():
    customer_id = request.args.get("customer_id")
    db = get_db()
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    if customer_id:
        c = db.execute("SELECT elevenlabs_agent_id FROM customers WHERE id=?", (customer_id,)).fetchone()
        if c and c["elevenlabs_agent_id"]:
            agent_id = c["elevenlabs_agent_id"]
    r = requests.get(
        "https://api.elevenlabs.io/v1/convai/conversation/get_signed_url",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        params={"agent_id": agent_id}
    )
    if r.status_code != 200:
        return jsonify({"error": r.text}), 500
    return jsonify(r.json())

@app.route("/avatar/thumbnail")
def avatar_thumbnail():
    customer_id = request.args.get("customer_id")
    db = get_db()
    customer = get_customer(db, customer_id) if customer_id else get_first_customer(db)
    if customer and customer["avatar_id"]:
        try:
            r = requests.get(
                f"https://api.liveavatar.com/v2/avatars/{customer['avatar_id']}",
                headers={"X-API-KEY": LIVEAVATAR_API_KEY}
            )
            if r.status_code == 200:
                d = r.json().get("data", {})
                thumb = d.get("thumbnail_url") or d.get("preview_url")
                if thumb:
                    img = requests.get(thumb)
                    return Response(img.content, mimetype=img.headers.get("Content-Type", "image/jpeg"))
        except Exception:
            pass
>>>>>>> f08164b (TRIONE digital twin — EGO engine, three portals, face recognition)
    local = os.path.join(os.path.dirname(__file__), "static", "petar.jpg")
    if os.path.exists(local):
        return send_file(local, mimetype="image/jpeg")
    return jsonify({"error": "No image"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
