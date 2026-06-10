import os, uuid, base64, hashlib, json, threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, g, session, redirect, url_for
from flask_cors import CORS
import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "trione-secret-change-in-prod")
CORS(app)

DATABASE_URL        = os.getenv("DATABASE_URL")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")          # Petar's cloned voice (voice ID only)
ELEVENLABS_MODEL    = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
RECORDINGS_DIR      = os.getenv("RECORDINGS_DIR", os.path.join(os.path.dirname(__file__), "recordings"))

os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ── DB ────────────────────────────────────────────────────────────────────────

def get_db():
    if "db" not in g:
        g.db = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        try: db.close()
        except: pass

def init_db():
    tables = [
        """CREATE TABLE IF NOT EXISTS ava_customers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, voice_id TEXT, photo_path TEXT,
            persona_summary TEXT, ego_model TEXT,
            created_at TEXT, active INTEGER DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS ava_users (
            id TEXT PRIMARY KEY, face_encoding TEXT, name TEXT DEFAULT 'Guest',
            password_hash TEXT, first_seen TEXT, last_seen TEXT, visit_count INTEGER DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS ava_sessions (
            id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, user_id TEXT,
            session_type TEXT NOT NULL, started_at TEXT, ended_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_messages (
            id SERIAL PRIMARY KEY, session_id TEXT NOT NULL, customer_id TEXT NOT NULL,
            user_id TEXT, speaker TEXT NOT NULL, text TEXT NOT NULL, timestamp TEXT NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS ava_knowledge_base (
            id SERIAL PRIMARY KEY, customer_id TEXT NOT NULL, chunk TEXT NOT NULL,
            source_session_id TEXT, category TEXT, ego_layer TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_ego_revisions (
            id SERIAL PRIMARY KEY, customer_id TEXT NOT NULL, revision TEXT NOT NULL,
            trigger_text TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_recordings (
            id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, session_id TEXT,
            filename TEXT NOT NULL, mime TEXT, size_bytes INTEGER,
            transcript TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_admins (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, created_at TEXT)""",
    ]
    for sql in tables:
        try:
            with psycopg2.connect(DATABASE_URL) as db:
                with db.cursor() as cur:
                    cur.execute(sql)
                db.commit()
        except Exception as e:
            print(f"Table create warning: {e}")

    # ── Migrations: add columns to tables that already existed from older versions ──
    migrations = [
        "ALTER TABLE ava_users ADD COLUMN IF NOT EXISTS password_hash TEXT",
        "ALTER TABLE ava_customers ADD COLUMN IF NOT EXISTS voice_id TEXT",
        "ALTER TABLE ava_customers ADD COLUMN IF NOT EXISTS photo_path TEXT",
    ]
    for sql in migrations:
        try:
            with psycopg2.connect(DATABASE_URL) as db:
                with db.cursor() as cur:
                    cur.execute(sql)
                db.commit()
        except Exception as e:
            print(f"Migration warning: {e}")

init_db()

# ── Seed admin ────────────────────────────────────────────────────────────────

def seed_admin():
    try:
        with psycopg2.connect(DATABASE_URL) as db:
            with db.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ava_admins")
                if cur.fetchone()[0] == 0:
                    cur.execute(
                        "INSERT INTO ava_admins (id,email,password_hash,created_at) VALUES (%s,%s,%s,%s)",
                        (str(uuid.uuid4()), os.getenv("ADMIN_EMAIL", "admin@example.com"),
                         hashlib.sha256(os.getenv("ADMIN_PASSWORD", "password").encode()).hexdigest(),
                         datetime.utcnow().isoformat())
                    )
            db.commit()
            print("Default admin seeded.")
    except Exception as e:
        print(f"Admin seed warning: {e}")

seed_admin()

# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_pw(pw): return hashlib.sha256((pw or "").encode()).hexdigest()

def get_customer(db, cid):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM ava_customers WHERE id=%s", (cid,))
        return cur.fetchone()

def get_first_customer(db):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM ava_customers WHERE active=1 ORDER BY created_at LIMIT 1")
        return cur.fetchone()

# ── Gemini (the brain) ─────────────────────────────────────────────────────────

def gemini(prompt, system=None):
    if not GEMINI_API_KEY: return ""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "Understood."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})
    try:
        r = requests.post(url, json={"contents": contents}, timeout=60)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Gemini error {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print(f"Gemini exception: {e}")
    return ""

def gemini_reply(persona_context, history, user_msg, speaker_name="Guest"):
    """Generate a reply AS the digital twin (Petar), grounded in his EGO."""
    convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in history[-12:])
    system = f"""You ARE this person speaking in first person. Stay fully in character.
Never say you are an AI or a model. Answer as the person would, using their beliefs,
values, memories and relationships below. If something is not in your knowledge,
respond as the person honestly would ("I don't recall that") rather than inventing facts.
Keep replies natural, spoken-length (1-4 sentences unless asked for more).

{persona_context if persona_context else "(No persona learned yet — speak warmly and ask the visitor to tell you about themselves.)"}"""
    prompt = f"""Recent conversation:
{convo}

{speaker_name} just said: "{user_msg}"

Reply now, in first person, as yourself:"""
    return gemini(prompt, system) or "I'm here. Tell me what's on your mind."

def gemini_interview_question(persona_context, history, last_answer):
    """The twin acts as a warm interviewer drawing out the person's life and identity."""
    convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in history[-10:])
    system = """You are a warm, curious interviewer helping a person build their digital twin.
Your goal is to draw out who they are — beliefs, memories, relationships, values, turning points.
Speak in first person to them, like a close friend who really wants to understand.
Ask ONE short, specific follow-up question at a time (1-2 sentences). React briefly to what
they just said, then ask the next question that goes deeper. Never list questions. Be spoken and natural."""
    prompt = f"""What they've told you so far:
{persona_context if persona_context else "(nothing yet)"}

Recent exchange:
{convo}

They just said: "{last_answer}"

Respond warmly in 1-2 sentences and ask your next single question:"""
    return gemini(prompt, system) or "Thank you for sharing that. Tell me more about a moment that shaped who you are."

def gemini_interview_opening(persona_context):
    system = "You are a warm interviewer starting a session to build someone's digital twin. Greet them in one or two friendly spoken sentences and ask an easy opening question to get them talking about themselves."
    prompt = f"What you already know about them:\n{persona_context or '(nothing yet — this is the first session)'}\n\nGive your spoken opening now:"
    return gemini(prompt, system) or "Hi — good to see you. Let's pick up where we left off. Tell me, what's been on your mind today?"

def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    try: return json.loads(text)
    except: return None

# ── ElevenLabs (voice ID only) ─────────────────────────────────────────────────

def elevenlabs_tts(text, voice_id=None):
    """Synthesize speech with the cloned voice ID. Returns (b64_mp3, error)."""
    vid = (voice_id or ELEVENLABS_VOICE_ID or "").strip()
    if not ELEVENLABS_API_KEY:
        return None, "Missing ELEVENLABS_API_KEY"
    if not vid:
        return None, "Missing ELEVENLABS_VOICE_ID"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            return base64.b64encode(r.content).decode(), None
        return None, f"ElevenLabs error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"ElevenLabs exception: {e}"

# ── EGO engine ──────────────────────────────────────────────────────────────────

def extract_ego(text, existing_ego_summary="", speaker_context="general conversation"):
    prompt = f"""You are building an Artificial EGO — a living model of a person's identity, reasoning and relationships.

Existing EGO summary: {existing_ego_summary}

New input: "{text}"
Context: {speaker_context}

Extract and return ONLY valid JSON, no markdown:
{{
  "beliefs": [],
  "contradictions": [],
  "relationships": {{}},
  "reasoning_patterns": [],
  "values": [],
  "raw_chunks": [{{"chunk": "...", "category": "belief|experience|preference|fact|story|relationship"}}]
}}"""
    raw = gemini(prompt)
    result = parse_json(raw)
    return result or {
        "beliefs": [], "contradictions": [], "relationships": {},
        "reasoning_patterns": [], "values": [],
        "raw_chunks": [{"chunk": text[:500], "category": "fact"}]
    }

def update_ego_model(customer_id, new_ego_data, trigger_text):
    try:
        with psycopg2.connect(DATABASE_URL) as db:
            db.autocommit = False
            with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT ego_model, persona_summary FROM ava_customers WHERE id=%s", (customer_id,))
                row = cur.fetchone()
                existing = {}
                if row and row["ego_model"]:
                    try: existing = json.loads(row["ego_model"])
                    except: pass

                for key in ["beliefs", "reasoning_patterns", "values"]:
                    existing[key] = list(set(existing.get(key, []) + new_ego_data.get(key, [])))

                rels = existing.get("relationships", {})
                for person, dynamic in new_ego_data.get("relationships", {}).items():
                    rels[person] = (rels.get(person, "") + " | " + dynamic).strip(" | ")
                existing["relationships"] = rels

                contradictions = existing.get("contradictions", [])
                contradictions.extend(new_ego_data.get("contradictions", []))
                existing["contradictions"] = contradictions[-20:]

                summary = gemini(f"Summarise this person's core identity in 3 sentences:\n{json.dumps(existing, indent=2)}")
                existing["summary"] = summary
                existing["last_updated"] = datetime.utcnow().isoformat()

                ego_json = json.dumps(existing)
                cur.execute("UPDATE ava_customers SET ego_model=%s, persona_summary=%s WHERE id=%s",
                            (ego_json, summary, customer_id))
                cur.execute(
                    "INSERT INTO ava_ego_revisions (customer_id, revision, trigger_text, created_at) VALUES (%s,%s,%s,%s)",
                    (customer_id, ego_json, trigger_text[:500], datetime.utcnow().isoformat())
                )
            db.commit()
            return summary
    except Exception as e:
        print(f"EGO update error: {e}")
    return ""

def get_ego_context(db, customer_id):
    with db.cursor() as cur:
        cur.execute("SELECT ego_model, persona_summary FROM ava_customers WHERE id=%s", (customer_id,))
        c = cur.fetchone()
        cur.execute(
            "SELECT chunk, category FROM ava_knowledge_base WHERE customer_id=%s ORDER BY created_at DESC LIMIT 40",
            (customer_id,)
        )
        chunks = cur.fetchall()

    ego = None
    if c and c["ego_model"]:
        try: ego = json.loads(c["ego_model"])
        except: pass

    parts = []
    if ego:
        if ego.get("summary"):            parts.append(f"=== PERSONA ===\n{ego['summary']}")
        if ego.get("beliefs"):            parts.append("=== BELIEFS ===\n" + "\n".join(f"• {b}" for b in ego["beliefs"][:10]))
        if ego.get("values"):             parts.append("=== VALUES ===\n" + "\n".join(f"• {v}" for v in ego["values"][:8]))
        if ego.get("reasoning_patterns"): parts.append("=== HOW THEY THINK ===\n" + "\n".join(f"• {r}" for r in ego["reasoning_patterns"][:6]))
        if ego.get("relationships"):
            parts.append("=== RELATIONSHIPS ===\n" + "\n".join(f"• {p}: {d}" for p, d in list(ego["relationships"].items())[:10]))
    if chunks:
        parts.append("=== KNOWLEDGE ===\n" + "\n".join(f"[{c['category']}] {c['chunk']}" for c in chunks))
    return "\n\n".join(parts)

# ── Face recognition ──────────────────────────────────────────────────────────

def encode_face(b64):
    try:
        import cv2, face_recognition
        if "," in b64: b64 = b64.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(b64), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        encs = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return encs[0] if encs else None
    except Exception as e:
        print(f"Face encode error: {e}"); return None

def find_face(db, enc, tolerance=0.5):
    try:
        import face_recognition
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_users WHERE face_encoding IS NOT NULL")
            for u in cur.fetchall():
                stored = np.array(json.loads(u["face_encoding"]))
                if face_recognition.face_distance([stored], enc)[0] < tolerance:
                    return dict(u)
    except Exception as e:
        print(f"Face match error: {e}")
    return None

# ════════════════════════════════════════════════════════════════════════════
# USER PORTAL  —  identified by face image + password, talks to Petar's twin
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def user_home():
    return render_template("user/index.html")

@app.route("/u/identify", methods=["POST"])
def user_identify():
    """image + password. New face -> register (needs name + password). Known face -> verify password."""
    data = request.get_json(silent=True) or {}
    b64 = data.get("image")
    password = data.get("password", "")
    name = (data.get("name") or "").strip()
    db = get_db()

    enc = encode_face(b64) if b64 else None
    if enc is None:
        return jsonify({"error": "No face detected. Center your face and try again."}), 400

    match = find_face(db, enc)

    if match:
        # returning user — verify password
        if not match.get("password_hash"):
            # legacy/no password set yet — set it now
            with db.cursor() as cur:
                cur.execute("UPDATE ava_users SET password_hash=%s, last_seen=%s, visit_count=visit_count+1 WHERE id=%s",
                            (hash_pw(password), datetime.utcnow().isoformat(), match["id"]))
            db.commit()
        elif match["password_hash"] != hash_pw(password):
            return jsonify({"error": "Password does not match this face.", "known": True}), 401
        else:
            with db.cursor() as cur:
                cur.execute("UPDATE ava_users SET last_seen=%s, visit_count=visit_count+1 WHERE id=%s",
                            (datetime.utcnow().isoformat(), match["id"]))
            db.commit()
        user_id, uname, returning = match["id"], match["name"], True
    else:
        # new user — register
        if not password:
            return jsonify({"error": "New here — set a password to remember you.", "register": True}), 200
        user_id = str(uuid.uuid4())
        uname = name or "Guest"
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO ava_users (id, face_encoding, name, password_hash, first_seen, last_seen) VALUES (%s,%s,%s,%s,%s,%s)",
                (user_id, json.dumps(enc.tolist()), uname, hash_pw(password),
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
            )
        db.commit()
        returning = False

    session["user_id"] = user_id

    with db.cursor() as cur:
        cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s ORDER BY timestamp DESC LIMIT 20", (user_id,))
        history = cur.fetchall()

    customer = get_first_customer(db)
    return jsonify({
        "ok": True, "user_id": user_id, "name": uname, "returning": returning,
        "avatar_name": customer["name"] if customer else "the avatar",
        "history": [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(history)]
    })

@app.route("/u/session", methods=["POST"])
def user_session():
    if "user_id" not in session:
        return jsonify({"error": "Not identified"}), 401
    db = get_db()
    customer = get_first_customer(db)
    if not customer:
        return jsonify({"error": "No active avatar found"}), 404

    session_id = str(uuid.uuid4())
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO ava_sessions (id, customer_id, user_id, session_type, started_at) VALUES (%s,%s,%s,%s,%s)",
            (session_id, customer["id"], session["user_id"], "user_chat", datetime.utcnow().isoformat())
        )
    # spoken greeting in the avatar's voice
    with db.cursor() as cur:
        cur.execute("SELECT name FROM ava_users WHERE id=%s", (session["user_id"],))
        urow = cur.fetchone()
    uname = urow["name"] if urow else "there"
    persona_context = get_ego_context(db, customer["id"])
    greeting = gemini_reply(persona_context, [], f"(A visitor named {uname} just arrived to talk with you. Greet them warmly in one or two sentences and invite them to talk.)", uname)
    db.commit()
    g_audio, g_err = elevenlabs_tts(greeting, customer.get("voice_id"))
    return jsonify({
        "session_id": session_id,
        "customer_id": customer["id"],
        "customer_name": customer["name"],
        "persona": customer["persona_summary"] or "",
        "greeting": greeting,
        "greeting_audio": g_audio,
        "voice_error": g_err
    })

@app.route("/u/talk", methods=["POST"])
def user_talk():
    """User message -> Gemini reply AS Petar -> ElevenLabs voice. Stores both turns."""
    if "user_id" not in session:
        return jsonify({"error": "Not identified"}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id")
    if not text:
        return jsonify({"error": "Empty message"}), 400

    db = get_db()
    customer = get_first_customer(db)
    if not customer:
        return jsonify({"error": "No active avatar"}), 404
    cid = customer["id"]
    uid = session["user_id"]

    # user name for context
    with db.cursor() as cur:
        cur.execute("SELECT name FROM ava_users WHERE id=%s", (uid,))
        row = cur.fetchone()
    speaker_name = row["name"] if row else "Guest"

    # history
    with db.cursor() as cur:
        cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s ORDER BY timestamp DESC LIMIT 12", (uid,))
        history = [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(cur.fetchall())]

    persona_context = get_ego_context(db, cid)
    reply = gemini_reply(persona_context, history, text, speaker_name)

    # store both turns
    now = datetime.utcnow().isoformat()
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_messages (session_id, customer_id, user_id, speaker, text, timestamp) VALUES (%s,%s,%s,%s,%s,%s)",
                    (session_id, cid, uid, "user", text, now))
        cur.execute("INSERT INTO ava_messages (session_id, customer_id, user_id, speaker, text, timestamp) VALUES (%s,%s,%s,%s,%s,%s)",
                    (session_id, cid, uid, "avatar", reply, datetime.utcnow().isoformat()))
    db.commit()

    audio_b64, voice_err = elevenlabs_tts(reply, customer.get("voice_id"))
    return jsonify({"reply": reply, "audio": audio_b64, "voice_error": voice_err})

# ════════════════════════════════════════════════════════════════════════════
# CUSTOMER PORTAL  —  Petar trains his twin: record + transcribe + learn
# ════════════════════════════════════════════════════════════════════════════

@app.route("/customer/login", methods=["GET", "POST"])
def customer_login():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_customers WHERE email=%s AND password_hash=%s",
                        (data.get("email"), hash_pw(data.get("password", ""))))
            c = cur.fetchone()
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
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_sessions (id, customer_id, session_type, started_at) VALUES (%s,%s,%s,%s)",
                    (session_id, customer["id"], "customer_training", datetime.utcnow().isoformat()))
        cur.execute("SELECT COUNT(*) as cnt FROM ava_knowledge_base WHERE customer_id=%s", (customer["id"],))
        knowledge_count = cur.fetchone()["cnt"]
    db.commit()

    ego = None
    if customer["ego_model"]:
        try: ego = json.loads(customer["ego_model"])
        except: pass

    persona_context = get_ego_context(db, customer["id"])
    opening = gemini_interview_opening(persona_context)
    o_audio, o_err = elevenlabs_tts(opening, customer.get("voice_id"))

    return jsonify({
        "session_id": session_id,
        "customer_id": customer["id"],
        "name": customer["name"],
        "knowledge_count": knowledge_count,
        "persona": customer["persona_summary"] or "",
        "ego_model": ego,
        "opening": opening,
        "opening_audio": o_audio,
        "voice_error": o_err
    })

@app.route("/customer/train", methods=["POST"])
def customer_train():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    session_id = data.get("session_id")
    if not text or len(text) < 5:
        return jsonify({"ok": True, "chunks_added": 0})

    db = get_db()
    customer_id = session["customer_id"]

    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_messages (session_id, customer_id, speaker, text, timestamp) VALUES (%s,%s,%s,%s,%s)",
                    (session_id, customer_id, "customer", text, datetime.utcnow().isoformat()))
    db.commit()

    customer = get_customer(db, customer_id)
    ego_data = extract_ego(text, customer["persona_summary"] or "", data.get("context", "conversation"))

    chunks_added = 0
    with db.cursor() as cur:
        for chunk in ego_data.get("raw_chunks", []):
            if chunk.get("chunk"):
                cur.execute(
                    "INSERT INTO ava_knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                    (customer_id, chunk["chunk"], session_id, chunk.get("category", "fact"), "raw", datetime.utcnow().isoformat()))
                chunks_added += 1
        for belief in ego_data.get("beliefs", []):
            cur.execute("INSERT INTO ava_knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                        (customer_id, belief, session_id, "belief", "ego", datetime.utcnow().isoformat()))
        for value in ego_data.get("values", []):
            cur.execute("INSERT INTO ava_knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                        (customer_id, value, session_id, "value", "ego", datetime.utcnow().isoformat()))
    db.commit()

    # learn synchronously so the persona updates reliably (no silent background failure)
    persona = update_ego_model(customer_id, ego_data, text) or (customer["persona_summary"] or "")

    # the twin speaks back: a warm follow-up question to go deeper
    with db.cursor() as cur:
        cur.execute("SELECT speaker, text FROM ava_messages WHERE customer_id=%s AND user_id IS NULL ORDER BY timestamp DESC LIMIT 10", (customer_id,))
        history = [{"speaker": ("you" if r["speaker"] == "customer" else "twin"), "text": r["text"]} for r in reversed(cur.fetchall())]
    persona_context = get_ego_context(db, customer_id)
    question = gemini_interview_question(persona_context, history, text)

    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_messages (session_id, customer_id, speaker, text, timestamp) VALUES (%s,%s,%s,%s,%s)",
                    (session_id, customer_id, "twin", question, datetime.utcnow().isoformat()))
    db.commit()

    q_audio, q_err = elevenlabs_tts(question, customer.get("voice_id"))

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) as cnt FROM ava_knowledge_base WHERE customer_id=%s", (customer_id,))
        total = cur.fetchone()["cnt"]

    return jsonify({
        "ok": True,
        "chunks_added": chunks_added,
        "beliefs_extracted": len(ego_data.get("beliefs", [])),
        "values_extracted": len(ego_data.get("values", [])),
        "knowledge_count": total,
        "persona": persona,
        "reply": question,
        "audio": q_audio,
        "voice_error": q_err
    })

@app.route("/customer/upload_recording", methods=["POST"])
def customer_upload_recording():
    """Receives the browser MediaRecorder blob (open-source camera). Stores to disk + DB."""
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    session_id = request.form.get("session_id", "")
    transcript = request.form.get("transcript", "")
    rec_id = str(uuid.uuid4())
    ext = "webm" if "webm" in (f.mimetype or "") else "bin"
    fname = f"{session['customer_id']}_{rec_id}.{ext}"
    path = os.path.join(RECORDINGS_DIR, fname)
    f.save(path)
    size = os.path.getsize(path)

    db = get_db()
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO ava_recordings (id, customer_id, session_id, filename, mime, size_bytes, transcript, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (rec_id, session["customer_id"], session_id, fname, f.mimetype, size, transcript, datetime.utcnow().isoformat()))
    db.commit()
    return jsonify({"ok": True, "recording_id": rec_id, "size_bytes": size})

@app.route("/customer/recordings")
def customer_recordings():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id, filename, mime, size_bytes, transcript, created_at FROM ava_recordings WHERE customer_id=%s ORDER BY created_at DESC LIMIT 50",
                    (session["customer_id"],))
        rows = cur.fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/customer/recording/<rid>")
def customer_recording_file(rid):
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT filename, mime FROM ava_recordings WHERE id=%s AND customer_id=%s", (rid, session["customer_id"]))
        r = cur.fetchone()
    if not r:
        return jsonify({"error": "Not found"}), 404
    return send_file(os.path.join(RECORDINGS_DIR, r["filename"]), mimetype=r["mime"] or "video/webm")

@app.route("/customer/knowledge")
def customer_knowledge():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id, chunk, category, ego_layer, created_at FROM ava_knowledge_base WHERE customer_id=%s ORDER BY created_at DESC",
                    (session["customer_id"],))
        chunks = cur.fetchall()
    return jsonify([dict(c) for c in chunks])

@app.route("/customer/knowledge/<int:kid>", methods=["DELETE"])
def delete_knowledge(kid):
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM ava_knowledge_base WHERE id=%s AND customer_id=%s", (kid, session["customer_id"]))
    db.commit()
    return jsonify({"ok": True})

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
    return jsonify({"name": c["name"], "email": c["email"], "persona": c["persona_summary"] or "", "ego": ego})

# ════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ════════════════════════════════════════════════════════════════════════════

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_admins WHERE email=%s AND password_hash=%s",
                        (data.get("email"), hash_pw(data.get("password", ""))))
            a = cur.fetchone()
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

@app.route("/admin/customers", methods=["GET", "POST"])
def admin_customers():
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        db = get_db()
        cid = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO ava_customers (id, name, email, password_hash, voice_id, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (cid, data["name"], data["email"], hash_pw(data["password"]),
                 (data.get("voice_id") or ELEVENLABS_VOICE_ID or "").strip(),
                 datetime.utcnow().isoformat()))
        db.commit()
        return jsonify({"ok": True, "id": cid})
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id, name, email, voice_id, active, created_at FROM ava_customers ORDER BY created_at DESC")
        rows = cur.fetchall()
    return jsonify([dict(c) for c in rows])

@app.route("/admin/customer/<cid>/update", methods=["POST"])
def admin_update_customer(cid):
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    db = get_db()
    with db.cursor() as cur:
        for field in ["voice_id"]:
            if field in data and data[field] is not None:
                cur.execute(f"UPDATE ava_customers SET {field}=%s WHERE id=%s", (data[field].strip(), cid))
    db.commit()
    return jsonify({"ok": True})

@app.route("/admin/toggle_customer/<cid>", methods=["POST"])
def admin_toggle_customer(cid):
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("UPDATE ava_customers SET active=1-active WHERE id=%s", (cid,))
    db.commit()
    return jsonify({"ok": True})

@app.route("/admin/stats")
def admin_stats():
    if "admin_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    stats = {}
    with db.cursor() as cur:
        for table, key in [
            ("ava_customers", "total_customers"), ("ava_users", "total_users"),
            ("ava_sessions", "total_sessions"), ("ava_messages", "total_messages"),
            ("ava_knowledge_base", "total_knowledge_chunks"),
            ("ava_recordings", "total_recordings"),
            ("ava_ego_revisions", "total_ego_revisions")
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[key] = cur.fetchone()["count"]
    return jsonify(stats)

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

# ── Avatar thumbnail ──────────────────────────────────────────────────────────

@app.route("/avatar/thumbnail")
def avatar_thumbnail():
    local = os.path.join(os.path.dirname(__file__), "static", "petar.jpg")
    if os.path.exists(local):
        return send_file(local, mimetype="image/jpeg")
    return jsonify({"error": "No image"}), 404

@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
