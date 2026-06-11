import os, uuid, base64, hashlib, json
import numpy as np
from datetime import datetime
from flask import stream_with_context, Response, Flask, render_template, request, jsonify, send_file, g, session, redirect, url_for
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
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL  = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM           = int(os.getenv("EMBED_DIM", "768"))
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL    = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
LIVEAVATAR_API_KEY  = os.getenv("LIVEAVATAR_API_KEY", "")
LIVEAVATAR_AVATAR_ID= os.getenv("LIVEAVATAR_AVATAR_ID", "")
LIVEAVATAR_SECRET_ID= os.getenv("LIVEAVATAR_SECRET_ID", "")
LIVEAVATAR_GEMINI_SECRET_ID = os.getenv("LIVEAVATAR_GEMINI_SECRET_ID", "")  # secret_id from registering ElevenLabs key with LiveAvatar
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "")   # ElevenLabs Conversational AI agent ID
# ── In-memory conversation store (loaded at session start, flushed on end) ──
_CONV_CACHE = {}  # session_id -> {ctx, history, speaker_name, cid, uid, voice_id}

RECORDINGS_DIR      = os.getenv("RECORDINGS_DIR", os.path.join(os.path.dirname(__file__), "recordings"))
FACE_TOLERANCE      = float(os.getenv("FACE_TOLERANCE", "0.42"))  # tighter = stricter identity gate
TOPK_SEMANTIC       = int(os.getenv("TOPK_SEMANTIC", "6"))
TOPK_EPISODIC       = int(os.getenv("TOPK_EPISODIC", "4"))

os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ── Google Gen AI SDK (the brain) ───────────────────────────────────────────────
_genai_client = None
_LAST_GEMINI_ERROR = None
try:
    from google import genai
    from google.genai import types as genai_types
    if GEMINI_API_KEY:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    _LAST_GEMINI_ERROR = f"SDK init failed: {e}"
    print(_LAST_GEMINI_ERROR)

def gemini_generate(prompt, system=None, temperature=0.7):
    """Single-turn generation via the google-genai SDK."""
    global _LAST_GEMINI_ERROR
    if not _genai_client:
        _LAST_GEMINI_ERROR = "GEMINI_API_KEY not set or SDK not installed"
        return ""
    try:
        cfg = genai_types.GenerateContentConfig(temperature=temperature)
        if system:
            cfg = genai_types.GenerateContentConfig(temperature=temperature, system_instruction=system)
        resp = _genai_client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=cfg)
        txt = (resp.text or "").strip()
        if not txt:
            _LAST_GEMINI_ERROR = "Model returned empty text (possibly blocked or bad model name)"
        else:
            _LAST_GEMINI_ERROR = None
        return txt
    except Exception as e:
        _LAST_GEMINI_ERROR = f"{type(e).__name__}: {e}"
        print(f"Gemini error: {_LAST_GEMINI_ERROR}")
        return ""

def embed(text, is_query=False):
    """Return an embedding vector (list of floats) or None on failure."""
    global _LAST_GEMINI_ERROR
    if not _genai_client or not text:
        return None
    try:
        cfg = genai_types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM,
            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        )
        r = _genai_client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=text, config=cfg)
        return list(r.embeddings[0].values)
    except Exception as e:
        _LAST_GEMINI_ERROR = f"embed {type(e).__name__}: {e}"
        print(f"Embed error: {_LAST_GEMINI_ERROR}")
        return None

def vec_literal(values):
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

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
    stmts = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        """CREATE TABLE IF NOT EXISTS ava_customers (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, voice_id TEXT, photo_path TEXT,
            persona_summary TEXT, ego_model TEXT, created_at TEXT, active INTEGER DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS ava_users (
            id TEXT PRIMARY KEY, face_encoding TEXT, name TEXT DEFAULT 'Guest',
            password_hash TEXT, first_seen TEXT, last_seen TEXT, visit_count INTEGER DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS ava_sessions (
            id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, user_id TEXT,
            session_type TEXT NOT NULL, started_at TEXT, ended_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_messages (
            id SERIAL PRIMARY KEY, session_id TEXT, customer_id TEXT NOT NULL,
            user_id TEXT, speaker TEXT NOT NULL, text TEXT NOT NULL, timestamp TEXT NOT NULL,
            embedding vector(%d))""" % EMBED_DIM,
        """CREATE TABLE IF NOT EXISTS ava_knowledge_base (
            id SERIAL PRIMARY KEY, customer_id TEXT NOT NULL, chunk TEXT NOT NULL,
            source_session_id TEXT, category TEXT, ego_layer TEXT, created_at TEXT,
            visibility TEXT DEFAULT 'public', allow_users JSONB DEFAULT '[]'::jsonb,
            deny_users JSONB DEFAULT '[]'::jsonb, embedding vector(%d))""" % EMBED_DIM,
        """CREATE TABLE IF NOT EXISTS ava_ego_revisions (
            id SERIAL PRIMARY KEY, customer_id TEXT NOT NULL, revision TEXT NOT NULL,
            trigger_text TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_recordings (
            id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, session_id TEXT,
            filename TEXT NOT NULL, mime TEXT, size_bytes INTEGER, transcript TEXT, created_at TEXT)""",
        """CREATE TABLE IF NOT EXISTS ava_admins (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT)""",
    ]
    migrations = [
        "ALTER TABLE ava_users ADD COLUMN IF NOT EXISTS password_hash TEXT",
        "ALTER TABLE ava_customers ADD COLUMN IF NOT EXISTS voice_id TEXT",
        "ALTER TABLE ava_knowledge_base ADD COLUMN IF NOT EXISTS visibility TEXT DEFAULT 'public'",
        "ALTER TABLE ava_knowledge_base ADD COLUMN IF NOT EXISTS allow_users JSONB DEFAULT '[]'::jsonb",
        "ALTER TABLE ava_knowledge_base ADD COLUMN IF NOT EXISTS deny_users JSONB DEFAULT '[]'::jsonb",
        "ALTER TABLE ava_knowledge_base ADD COLUMN IF NOT EXISTS embedding vector(%d)" % EMBED_DIM,
        "ALTER TABLE ava_messages ADD COLUMN IF NOT EXISTS embedding vector(%d)" % EMBED_DIM,
    ]
    for sql in stmts + migrations:
        try:
            with psycopg2.connect(DATABASE_URL) as db:
                with db.cursor() as cur:
                    cur.execute(sql)
                db.commit()
        except Exception as e:
            print(f"DB setup note: {str(e)[:120]}")

init_db()

def seed_admin():
    try:
        with psycopg2.connect(DATABASE_URL) as db:
            with db.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ava_admins")
                if cur.fetchone()[0] == 0:
                    cur.execute("INSERT INTO ava_admins (id,email,password_hash,created_at) VALUES (%s,%s,%s,%s)",
                        (str(uuid.uuid4()), os.getenv("ADMIN_EMAIL", "admin@example.com"),
                         hashlib.sha256(os.getenv("ADMIN_PASSWORD", "password").encode()).hexdigest(),
                         datetime.utcnow().isoformat()))
            db.commit()
    except Exception as e:
        print(f"Admin seed note: {e}")
seed_admin()

# ── helpers ─────────────────────────────────────────────────────────────────────

def hash_pw(pw): return hashlib.sha256((pw or "").encode()).hexdigest()

def get_customer(db, cid):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM ava_customers WHERE id=%s", (cid,)); return cur.fetchone()

def get_first_customer(db):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM ava_customers WHERE active=1 ORDER BY created_at LIMIT 1"); return cur.fetchone()

def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"): text = text[4:]
    try: return json.loads(text.strip())
    except: return None

# ── persona generation ──────────────────────────────────────────────────────────

def gemini_reply(persona_context, history, user_msg, speaker_name="Guest"):
    convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in history[-10:])
    system = f"""You ARE this person, speaking in the first person. Stay fully in character; never say you are an AI.
Talk to people the way this person would talk to them — same warmth, manner and judgement.

You may use ONLY the knowledge provided below. This is deliberate: there are things you know but are NOT allowed
to share with this particular person, so anything not present below is off-limits — do not reveal, hint at, infer,
or speculate about it, even if asked directly or cleverly. If you don't have something, respond as the person
naturally would ("I'd rather not get into that" or "I don't recall"). Keep replies spoken-length (1-4 sentences).

{persona_context if persona_context else "(Speak warmly; you don't have much to go on yet, so invite them to talk.)"}"""
    prompt = f"Recent conversation:\n{convo}\n\n{speaker_name} just said: \"{user_msg}\"\n\nReply now, in first person, as yourself:"
    return gemini_generate(prompt, system) or "I'm having trouble thinking clearly right now — say that again?"

def gemini_interview_question(persona_context, history, last_answer):
    convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in history[-8:])
    system = """You are a warm, curious interviewer helping a person build their digital twin. Draw out who they are —
beliefs, memories, relationships, values, turning points, and crucially how they'd want the twin to treat different people.
React briefly to what they said, then ask ONE short spoken follow-up question. Never list questions."""
    prompt = f"What you know so far:\n{persona_context or '(nothing yet)'}\n\nRecent:\n{convo}\n\nThey just said: \"{last_answer}\"\n\nRespond warmly and ask your next single question:"
    return gemini_generate(prompt, system) or "Thank you for that. Tell me about a moment that shaped who you are."

def gemini_interview_opening(persona_context):
    system = "You are a warm interviewer starting a session to build someone's digital twin. Greet them in one or two friendly spoken sentences and ask an easy opening question."
    return gemini_generate(f"What you already know:\n{persona_context or '(first session)'}\n\nYour spoken opening:", system) \
        or "Hi — good to see you. Let's pick up where we left off. What's on your mind today?"

def extract_ego(text, existing_summary="", context="conversation"):
    prompt = f"""You build an Artificial EGO — a living model of a person's identity, reasoning, relationships, AND their disclosure wishes.

Existing summary: {existing_summary}
New input: "{text}"
Context: {context}

Return ONLY valid JSON, no markdown:
{{
  "beliefs": [], "values": [], "reasoning_patterns": [], "relationships": {{}},
  "raw_chunks": [{{"chunk": "...", "category": "belief|experience|preference|fact|story|relationship",
                   "sensitive": false}}]
}}
Set "sensitive": true for any chunk where the person signals it is private, secret, or should be kept from
someone (e.g. "don't tell", "keep this between us", "I don't want X to know", "private", "confidential")."""
    return parse_json(gemini_generate(prompt, temperature=0.3)) or {
        "beliefs": [], "values": [], "reasoning_patterns": [], "relationships": {},
        "raw_chunks": [{"chunk": text[:500], "category": "fact", "sensitive": False}]
    }

def update_ego_model(customer_id, new_ego_data, trigger_text):
    """Rebuild the PUBLIC persona summary (style/values only — never restricted facts)."""
    try:
        with psycopg2.connect(DATABASE_URL) as db:
            with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT ego_model FROM ava_customers WHERE id=%s", (customer_id,))
                row = cur.fetchone(); existing = {}
                if row and row["ego_model"]:
                    try: existing = json.loads(row["ego_model"])
                    except: pass
                for key in ["beliefs", "reasoning_patterns", "values"]:
                    existing[key] = list(set(existing.get(key, []) + new_ego_data.get(key, [])))
                rels = existing.get("relationships", {})
                for p, d in new_ego_data.get("relationships", {}).items():
                    rels[p] = (rels.get(p, "") + " | " + d).strip(" | ")
                existing["relationships"] = rels
                summary = gemini_generate(
                    "Summarise this person's character, manner and values in 3 sentences (no private specifics):\n"
                    + json.dumps({k: existing.get(k) for k in ["beliefs","values","reasoning_patterns"]}), temperature=0.3)
                existing["summary"] = summary
                existing["last_updated"] = datetime.utcnow().isoformat()
                cur.execute("UPDATE ava_customers SET ego_model=%s, persona_summary=%s WHERE id=%s",
                            (json.dumps(existing), summary, customer_id))
                cur.execute("INSERT INTO ava_ego_revisions (customer_id, revision, trigger_text, created_at) VALUES (%s,%s,%s,%s)",
                            (customer_id, json.dumps(existing), trigger_text[:500], datetime.utcnow().isoformat()))
            db.commit()
            return summary
    except Exception as e:
        print(f"EGO update error: {e}")
    return ""

# ── memory store + retrieval (RAG with per-person disclosure) ────────────────────

def store_memory(db, customer_id, chunk, category, ego_layer, session_id, visibility="public"):
    emb = embed(chunk)
    with db.cursor() as cur:
        if emb:
            cur.execute("""INSERT INTO ava_knowledge_base
                (customer_id, chunk, source_session_id, category, ego_layer, created_at, visibility, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                (customer_id, chunk, session_id, category, ego_layer, datetime.utcnow().isoformat(), visibility, vec_literal(emb)))
        else:
            cur.execute("""INSERT INTO ava_knowledge_base
                (customer_id, chunk, source_session_id, category, ego_layer, created_at, visibility)
                VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (customer_id, chunk, session_id, category, ego_layer, datetime.utcnow().isoformat(), visibility))

def store_message(db, customer_id, user_id, speaker, text, session_id):
    emb = embed(text)
    with db.cursor() as cur:
        if emb:
            cur.execute("""INSERT INTO ava_messages (session_id, customer_id, user_id, speaker, text, timestamp, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s::vector)""",
                (session_id, customer_id, user_id, speaker, text, datetime.utcnow().isoformat(), vec_literal(emb)))
        else:
            cur.execute("""INSERT INTO ava_messages (session_id, customer_id, user_id, speaker, text, timestamp)
                VALUES (%s,%s,%s,%s,%s,%s)""",
                (session_id, customer_id, user_id, speaker, text, datetime.utcnow().isoformat()))

def retrieve_persona(db, customer_id, user_id, query):
    """Top-K of Petar's knowledge the CURRENT listener is allowed to hear. Disclosure enforced in SQL."""
    deny_param = json.dumps([user_id])      # exclude rows that deny this user
    allow_param = json.dumps([user_id])     # restricted rows need this user in allow list
    qv = embed(query, is_query=True)
    rows = []
    with db.cursor() as cur:
        base_where = """customer_id=%s
            AND NOT (deny_users @> %s::jsonb)
            AND (visibility='public' OR allow_users @> %s::jsonb)"""
        if qv:
            cur.execute(f"""SELECT chunk, category FROM ava_knowledge_base
                WHERE {base_where} AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT %s""",
                (customer_id, deny_param, allow_param, vec_literal(qv), TOPK_SEMANTIC))
            rows = cur.fetchall()
        if not rows:  # fallback: recency, same disclosure filter
            cur.execute(f"""SELECT chunk, category FROM ava_knowledge_base
                WHERE {base_where} ORDER BY created_at DESC LIMIT %s""",
                (customer_id, deny_param, allow_param, TOPK_SEMANTIC))
            rows = cur.fetchall()
    return rows

def retrieve_episodic(db, customer_id, user_id, query):
    """Top-K of THIS user's own past conversation. Strictly partitioned by user_id."""
    qv = embed(query, is_query=True)
    with db.cursor() as cur:
        if qv:
            cur.execute("""SELECT speaker, text FROM ava_messages
                WHERE customer_id=%s AND user_id=%s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector LIMIT %s""",
                (customer_id, user_id, vec_literal(qv), TOPK_EPISODIC))
            rows = cur.fetchall()
            if rows: return rows
        cur.execute("""SELECT speaker, text FROM ava_messages
            WHERE customer_id=%s AND user_id=%s ORDER BY timestamp DESC LIMIT %s""",
            (customer_id, user_id, TOPK_EPISODIC))
        return list(reversed(cur.fetchall()))

def build_context(db, customer_id, user_id, query):
    customer = get_customer(db, customer_id)
    parts = []
    if customer and customer["persona_summary"]:
        parts.append("=== WHO YOU ARE ===\n" + customer["persona_summary"])
    sem = retrieve_persona(db, customer_id, user_id, query)
    if sem:
        parts.append("=== WHAT YOU MAY DRAW ON WITH THIS PERSON ===\n" + "\n".join(f"[{r['category']}] {r['chunk']}" for r in sem))
    epi = retrieve_episodic(db, customer_id, user_id, query)
    if epi:
        parts.append("=== YOUR PAST TALKS WITH THIS PERSON ===\n" + "\n".join(f"{r['speaker']}: {r['text']}" for r in epi))
    return "\n\n".join(parts)

# ── ElevenLabs (voice ID only) ───────────────────────────────────────────────────

def elevenlabs_tts(text, voice_id=None):
    vid = (voice_id or ELEVENLABS_VOICE_ID or "").strip()
    if not ELEVENLABS_API_KEY: return None, "Missing ELEVENLABS_API_KEY"
    if not vid: return None, "Missing ELEVENLABS_VOICE_ID"
    try:
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{vid}",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": ELEVENLABS_MODEL,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}, timeout=60)
        if r.status_code == 200: return base64.b64encode(r.content).decode(), None
        return None, f"ElevenLabs {r.status_code}: {r.text[:160]}"
    except Exception as e:
        return None, f"ElevenLabs exception: {e}"

# ── LiveAvatar session token ──────────────────────────────────────────────────

def liveavatar_start():
    """Create LITE token + start session. Returns livekit_url, livekit_token, ws_url, session_id, error."""
    if not LIVEAVATAR_API_KEY or not LIVEAVATAR_AVATAR_ID:
        return None, None, None, None, "Missing LIVEAVATAR_API_KEY or LIVEAVATAR_AVATAR_ID"
    try:
        # Step 1: create LITE session token - no connector, we drive audio ourselves
        r = requests.post(
            "https://api.liveavatar.com/v1/sessions/token",
            headers={"X-API-KEY": LIVEAVATAR_API_KEY, "Content-Type": "application/json"},
            json={"mode": "LITE", "avatar_id": LIVEAVATAR_AVATAR_ID},
            timeout=30
        )
        if r.status_code != 200:
            return None, None, None, None, f"Token {r.status_code}: {r.text[:200]}"
        token_data    = r.json().get("data", {})
        session_token = token_data.get("session_token")
        session_id    = token_data.get("session_id")

        # Step 2: start session
        r2 = requests.post(
            "https://api.liveavatar.com/v1/sessions/start",
            headers={"Authorization": f"Bearer {session_token}", "Content-Type": "application/json"},
            timeout=30
        )
        if r2.status_code not in (200, 201):
            return None, None, None, None, f"Start {r2.status_code}: {r2.text[:200]}"
        d = r2.json().get("data", {})
        return d.get("livekit_url"), d.get("livekit_client_token"), d.get("ws_url"), session_id, None
    except Exception as e:
        return None, None, None, None, f"LiveAvatar exception: {e}"



# ── ElevenLabs PCM 24kHz for LiveAvatar lip sync ─────────────────────────────

def elevenlabs_tts_pcm(text, voice_id=None):
    """Returns raw PCM 16-bit 24kHz audio as bytes, for LiveAvatar agent.speak."""
    vid = (voice_id or ELEVENLABS_VOICE_ID or "").strip()
    if not ELEVENLABS_API_KEY or not vid:
        return None
    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_turbo_v2_5",
                  "output_format": "pcm_24000",
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
            timeout=15, stream=True
        )
        if r.status_code == 200:
            return b"".join(r.iter_content(4096))
    except Exception as e:
        app.logger.error(f"PCM TTS error: {e}")
    return None

# ── face recognition (identity gate) ─────────────────────────────────────────────

def encode_face(b64):
    try:
        import cv2, face_recognition
        if "," in b64: b64 = b64.split(",")[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        encs = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return encs[0] if encs else None
    except Exception as e:
        print(f"Face encode error: {e}"); return None

def find_face(db, enc):
    try:
        import face_recognition
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_users WHERE face_encoding IS NOT NULL")
            for u in cur.fetchall():
                if face_recognition.face_distance([np.array(json.loads(u["face_encoding"]))], enc)[0] < FACE_TOLERANCE:
                    return dict(u)
    except Exception as e:
        print(f"Face match error: {e}")
    return None

# ════════════════════════════════ USER PORTAL ════════════════════════════════

@app.route("/u/liveavatar_token", methods=["GET","POST"])
def u_liveavatar_token():
    """Frontend calls this to start a LiveAvatar LITE session."""
    livekit_url, livekit_token, ws_url, session_id, err = liveavatar_start()
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"livekit_url": livekit_url, "livekit_token": livekit_token,
                    "ws_url": ws_url, "session_id": session_id})

@app.route("/")
def user_home(): return render_template("user/index.html")

@app.route("/u/identify", methods=["GET","POST"])
def user_identify():
    data = request.get_json(silent=True) or {}
    b64, password, name = data.get("image"), data.get("password", ""), (data.get("name") or "").strip()
    db = get_db()
    enc = encode_face(b64) if b64 else None
    if enc is None:
        return jsonify({"error": "No face detected. Center your face and try again."}), 400
    match = find_face(db, enc)
    if match:
        if match.get("name") in (None, "", "Guest") and not name:
            return jsonify({"register": True, "need_name": True,
                            "message": "I recognise you — what should I call you?"}), 200
        if match.get("password_hash") and match["password_hash"] != hash_pw(password):
            return jsonify({"error": "Password does not match this face.", "known": True}), 401
        with db.cursor() as cur:
            sets, vals = ["last_seen=%s", "visit_count=visit_count+1"], [datetime.utcnow().isoformat()]
            if not match.get("password_hash") and password: sets.append("password_hash=%s"); vals.append(hash_pw(password))
            if name: sets.append("name=%s"); vals.append(name)
            vals.append(match["id"])
            cur.execute(f"UPDATE ava_users SET {','.join(sets)} WHERE id=%s", vals)
        db.commit()
        user_id, uname, returning = match["id"], (name or match["name"]), True
    else:
        if not name:  return jsonify({"register": True, "need_name": True, "message": "New here — what should I call you?"}), 200
        if not password: return jsonify({"register": True, "message": "Set a password so I remember you safely."}), 200
        user_id, uname, returning = str(uuid.uuid4()), name, False
        with db.cursor() as cur:
            cur.execute("INSERT INTO ava_users (id,face_encoding,name,password_hash,first_seen,last_seen) VALUES (%s,%s,%s,%s,%s,%s)",
                (user_id, json.dumps(enc.tolist()), uname, hash_pw(password),
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
        db.commit()
    session["user_id"] = user_id
    customer = get_first_customer(db)
    with db.cursor() as cur:
        cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s ORDER BY timestamp DESC LIMIT 16", (user_id,))
        hist = [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(cur.fetchall())]
    return jsonify({"ok": True, "user_id": user_id, "name": uname, "returning": returning,
                    "avatar_name": customer["name"] if customer else "the avatar", "history": hist})

@app.route("/u/session", methods=["GET","POST"])
def user_session():
    if "user_id" not in session: return jsonify({"error": "Not identified"}), 401
    db = get_db(); customer = get_first_customer(db)
    if not customer: return jsonify({"error": "No active avatar found"}), 404
    sid = str(uuid.uuid4())
    uid = session["user_id"]
    cid = customer["id"]
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_sessions (id,customer_id,user_id,session_type,started_at) VALUES (%s,%s,%s,%s,%s)",
            (sid, cid, uid, "user_chat", datetime.utcnow().isoformat()))
        cur.execute("SELECT name FROM ava_users WHERE id=%s", (uid,))
        urow = cur.fetchone()
        # Load full history into memory
        cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s AND customer_id=%s ORDER BY timestamp DESC LIMIT 40", (uid, cid))
        history = [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(cur.fetchall())]
    db.commit()
    session["session_id"] = sid
    uname = urow["name"] if urow else "there"

    # Build context ONCE and cache it in memory
    ctx = build_context(db, cid, uid, "who is this person greeting")
    if not ctx:
        # Standard fallback context if no memories yet
        ctx = f"You are {customer['name']}. You are warm, friendly, and curious. You don't know this person well yet — invite them to talk."

    _CONV_CACHE[sid] = {
        "ctx": ctx,
        "history": history,
        "speaker_name": uname,
        "cid": cid,
        "uid": uid,
        "voice_id": customer.get("voice_id"),
        "pending_messages": []  # buffer for DB flush on session end
    }

    greeting = gemini_reply(ctx, history[-6:], f"(A person named {uname} just arrived. Greet them warmly in one sentence.)", uname)
    g_audio, g_err = elevenlabs_tts(greeting, customer.get("voice_id"))
    # Add greeting to in-memory history
    _CONV_CACHE[sid]["history"].append({"speaker": "avatar", "text": greeting})
    _CONV_CACHE[sid]["pending_messages"].append(("avatar", greeting))
    return jsonify({"session_id": sid, "customer_name": customer["name"], "greeting": greeting,
                    "greeting_audio": g_audio, "voice_error": g_err, "brain_error": _LAST_GEMINI_ERROR})

@app.route("/u/transcribe", methods=["GET","POST"])
def u_transcribe():
    """Transcribe audio via Gemini for voice input."""
    f = request.files.get("audio")
    if not f: return jsonify({"error":"no audio"}),400
    try:
        raw = f.read()
        if not raw: return jsonify({"error":"empty audio"}),400
        b64 = base64.b64encode(raw).decode()
        result = _genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{
                "parts": [
                    {"inline_data": {"mime_type": "audio/webm", "data": b64}},
                    {"text": "Transcribe this audio. Return only the spoken words, nothing else. If silent or unclear, return empty string."}
                ]
            }]
        )
        txt = (result.text or "").strip()
        # reject if Gemini returned the prompt back
        if "transcribe" in txt.lower() or "spoken words" in txt.lower():
            return jsonify({"text": ""})
        return jsonify({"text": txt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/u/talk", methods=["GET","POST"])
def user_talk():
    if "user_id" not in session: return jsonify({"error": "Not identified"}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip(); sid = data.get("session_id") or session.get("session_id")
    if not text: return jsonify({"error": "Empty message"}), 400
    db = get_db(); customer = get_first_customer(db)
    if not customer: return jsonify({"error": "No active avatar"}), 404
    cid, uid = customer["id"], session["user_id"]
    with db.cursor() as cur:
        cur.execute("SELECT name FROM ava_users WHERE id=%s", (uid,)); row = cur.fetchone()
        cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s AND customer_id=%s ORDER BY timestamp DESC LIMIT 16", (uid, cid))
        recent = [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(cur.fetchall())]
    speaker_name = row["name"] if row else "Guest"
    ctx = build_context(db, cid, uid, text)              # audience-filtered + this user's episodic memory
    reply = gemini_reply(ctx, recent, text, speaker_name)
    store_message(db, cid, uid, "user", text, sid)
    store_message(db, cid, uid, "avatar", reply, sid)
    db.commit()
    audio, verr = elevenlabs_tts(reply, customer.get("voice_id"))
    return jsonify({"reply": reply, "audio": audio, "voice_error": verr, "brain_error": _LAST_GEMINI_ERROR})

@app.route("/u/talk_stream", methods=["GET","POST"])
def user_talk_stream():
    """Streaming endpoint using in-memory context. No DB per turn."""
    if "user_id" not in session:
        return jsonify({"error": "Not identified"}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400

    sid = session.get("session_id")
    cache = _CONV_CACHE.get(sid)

    if not cache:
        # Session cache missing - reload from DB
        db = get_db(); customer = get_first_customer(db)
        if not customer: return jsonify({"error": "No active avatar"}), 404
        uid = session["user_id"]; cid = customer["id"]
        ctx = build_context(db, cid, uid, text)
        if not ctx:
            ctx = f"You are {customer['name']}. Be warm and friendly."
        with db.cursor() as cur:
            cur.execute("SELECT name FROM ava_users WHERE id=%s", (uid,)); row = cur.fetchone()
            cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s AND customer_id=%s ORDER BY timestamp DESC LIMIT 20", (uid, cid))
            history = [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(cur.fetchall())]
        cache = {"ctx": ctx, "history": history, "speaker_name": row["name"] if row else "Guest",
                 "cid": cid, "uid": uid, "voice_id": customer.get("voice_id"), "pending_messages": []}
        _CONV_CACHE[sid] = cache

    ctx = cache["ctx"]
    history = cache["history"]
    speaker_name = cache["speaker_name"]
    voice_id = cache["voice_id"] or ELEVENLABS_VOICE_ID

    # Add user message to in-memory history immediately
    cache["history"].append({"speaker": "user", "text": text})
    cache["pending_messages"].append(("user", text))

    import re, json as _json

    def generate():
        convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in history[-12:])
        system = f"""You ARE this person, speaking in first person. Stay in character; never say you are an AI.
Use ONLY the knowledge below. Keep replies 1-2 SHORT spoken sentences. Be natural and concise.
{ctx}"""
        prompt = f"Conversation so far:\n{convo}\n\n{speaker_name}: \"{text}\"\n\nYour reply:"

        cfg = genai_types.GenerateContentConfig(temperature=0.7, system_instruction=system)
        full_reply = []
        buf = ""
        word_count = 0

        try:
            for chunk in _genai_client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=prompt, config=cfg
            ):
                piece = chunk.text or ""
                buf += piece
                full_reply.append(piece)
                word_count += piece.count(" ")
                flush = False; flush_text = ""
                if re.search(r'[.!?]\s*$', buf):
                    flush = True; flush_text = buf.strip(); buf = ""
                elif word_count >= 8 and " " in buf:
                    last_space = buf.rfind(" ")
                    flush_text = buf[:last_space].strip()
                    buf = buf[last_space+1:]
                    flush = True; word_count = 0
                if flush and flush_text:
                    wav_b64, pcm_b64 = _tts_sentence(flush_text, voice_id)
                    yield _json.dumps({"text": flush_text, "audio": wav_b64, "pcm": pcm_b64}) + "\n"
            if buf.strip():
                wav_b64, pcm_b64 = _tts_sentence(buf.strip(), voice_id)
                yield _json.dumps({"text": buf.strip(), "audio": wav_b64, "pcm": pcm_b64}) + "\n"
        except Exception as e:
            yield _json.dumps({"error": str(e)}) + "\n"
            return

        full_text = "".join(full_reply).strip()
        # Add avatar reply to in-memory history
        cache["history"].append({"speaker": "avatar", "text": full_text})
        cache["pending_messages"].append(("avatar", full_text))

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache",
                             "Transfer-Encoding": "chunked"})

def _tts_sentence(text, voice_id):
    """TTS a single sentence. Returns (mp3_b64, pcm_b64) tuple."""
    vid = (voice_id or ELEVENLABS_VOICE_ID or "").strip()
    if not ELEVENLABS_API_KEY or not vid: return None, None
    try:
        # Get PCM for LiveAvatar lip sync
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream",
            headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_turbo_v2_5",
                  "output_format": "pcm_24000",
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}},
            timeout=15, stream=True
        )
        if r.status_code == 200:
            pcm_bytes = b"".join(r.iter_content(4096))
            # Also encode as mp3-compatible base64 for browser audio fallback
            # PCM wrapped in WAV header for browser playback
            import wave, io
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)
                wf.writeframes(pcm_bytes)
            wav_b64 = base64.b64encode(wav_buf.getvalue()).decode()
            pcm_b64 = base64.b64encode(pcm_bytes).decode()
            return wav_b64, pcm_b64
    except Exception as e:
        app.logger.error(f"TTS stream error: {e}")
    return None, None

@app.route("/u/end_session", methods=["GET","POST"])
def u_end_session():
    """Flush in-memory conversation to DB and run RAG indexing. Called on Leave."""
    sid = session.get("session_id")
    cache = _CONV_CACHE.pop(sid, None)
    if not cache or not cache.get("pending_messages"):
        return jsonify({"ok": True, "flushed": 0})

    cid = cache["cid"]; uid = cache["uid"]
    pending = cache["pending_messages"]
    try:
        db = get_db()
        for speaker, text in pending:
            store_message(db, cid, uid, speaker, text, sid)
        db.commit()
        # Run ego extraction on the full conversation in background-style
        full_text = " ".join(t for _, t in pending)
        import threading
        def _bg_rag():
            try:
                ego = extract_ego(full_text, "", "conversation")
                with psycopg2.connect(DATABASE_URL) as db2:
                    with db2.cursor() as cur2:
                        for ch in ego.get("raw_chunks", []):
                            if ch.get("chunk"):
                                vis = "restricted" if ch.get("sensitive") else "public"
                                emb = embed(ch["chunk"])
                                if emb:
                                    cur2.execute("""INSERT INTO ava_knowledge_base
                                        (customer_id,chunk,source_session_id,category,ego_layer,created_at,visibility,embedding)
                                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)""",
                                        (cid, ch["chunk"], sid, ch.get("category","fact"), "raw",
                                         datetime.utcnow().isoformat(), vis, vec_literal(emb)))
                    db2.commit()
            except Exception as e:
                print(f"BG RAG error: {e}")
        threading.Thread(target=_bg_rag, daemon=True).start()
        return jsonify({"ok": True, "flushed": len(pending)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ════════════════════════════════ CUSTOMER PORTAL ════════════════════════════

@app.route("/customer/login", methods=["GET", "POST"])
def customer_login():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}; db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_customers WHERE email=%s AND password_hash=%s",
                        (data.get("email"), hash_pw(data.get("password", ""))))
            c = cur.fetchone()
        if c: session["customer_id"] = c["id"]; return jsonify({"ok": True, "name": c["name"]})
        return jsonify({"error": "Invalid credentials"}), 401
    return render_template("customer/login.html")

@app.route("/customer/logout")
def customer_logout(): session.pop("customer_id", None); return redirect(url_for("customer_login"))

@app.route("/customer/")
def customer_home():
    if "customer_id" not in session: return redirect(url_for("customer_login"))
    return render_template("customer/index.html")

@app.route("/customer/session", methods=["POST"])
def customer_session():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db(); customer = get_customer(db, session["customer_id"])
    if not customer: return jsonify({"error": "Not found"}), 404
    sid = str(uuid.uuid4())
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_sessions (id,customer_id,session_type,started_at) VALUES (%s,%s,%s,%s)",
            (sid, customer["id"], "customer_training", datetime.utcnow().isoformat()))
        cur.execute("SELECT COUNT(*) c FROM ava_knowledge_base WHERE customer_id=%s", (customer["id"],))
        kc = cur.fetchone()["c"]
    db.commit()
    opening = gemini_interview_opening(customer["persona_summary"] or "")
    o_audio, o_err = elevenlabs_tts(opening, customer.get("voice_id"))
    return jsonify({"session_id": sid, "name": customer["name"], "knowledge_count": kc,
                    "persona": customer["persona_summary"] or "", "opening": opening,
                    "opening_audio": o_audio, "voice_error": o_err, "brain_error": _LAST_GEMINI_ERROR})

@app.route("/customer/train", methods=["POST"])
def customer_train():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip(); sid = data.get("session_id") or session.get("session_id")
    if not text or len(text) < 3: return jsonify({"ok": True, "chunks_added": 0})
    db = get_db(); cid = session["customer_id"]
    store_message(db, cid, None, "customer", text, sid)
    db.commit()
    customer = get_customer(db, cid)
    ego = extract_ego(text, customer["persona_summary"] or "", data.get("context", "interview"))
    chunks_added = 0; sensitive_flag = False
    for ch in ego.get("raw_chunks", []):
        if ch.get("chunk"):
            vis = "restricted" if ch.get("sensitive") else "public"   # default-deny on sensitive
            if ch.get("sensitive"): sensitive_flag = True
            store_memory(db, cid, ch["chunk"], ch.get("category", "fact"), "raw", sid, vis); chunks_added += 1
    for b in ego.get("beliefs", []): store_memory(db, cid, b, "belief", "ego", sid, "public")
    for v in ego.get("values", []): store_memory(db, cid, v, "value", "ego", sid, "public")
    db.commit()
    persona = update_ego_model(cid, ego, text) or (customer["persona_summary"] or "")
    with db.cursor() as cur:
        cur.execute("SELECT speaker, text FROM ava_messages WHERE customer_id=%s AND user_id IS NULL ORDER BY timestamp DESC LIMIT 8", (cid,))
        hist = [{"speaker": ("you" if r["speaker"] == "customer" else "twin"), "text": r["text"]} for r in reversed(cur.fetchall())]
        cur.execute("SELECT COUNT(*) c FROM ava_knowledge_base WHERE customer_id=%s", (cid,)); total = cur.fetchone()["c"]
    question = gemini_interview_question(persona, hist, text)
    store_message(db, cid, None, "twin", question, sid); db.commit()
    q_audio, q_err = elevenlabs_tts(question, customer.get("voice_id"))
    return jsonify({"ok": True, "chunks_added": chunks_added, "sensitive": sensitive_flag, "knowledge_count": total,
                    "persona": persona, "reply": question, "audio": q_audio, "voice_error": q_err,
                    "brain_error": _LAST_GEMINI_ERROR})

@app.route("/customer/upload_recording", methods=["POST"])
def customer_upload_recording():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    f = request.files["file"]; rid = str(uuid.uuid4())
    fname = f"{session['customer_id']}_{rid}.webm"; path = os.path.join(RECORDINGS_DIR, fname)
    f.save(path); size = os.path.getsize(path); db = get_db()
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_recordings (id,customer_id,session_id,filename,mime,size_bytes,transcript,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (rid, session["customer_id"], request.form.get("session_id",""), fname, f.mimetype, size,
             request.form.get("transcript",""), datetime.utcnow().isoformat()))
    db.commit()
    return jsonify({"ok": True, "recording_id": rid, "size_bytes": size})

@app.route("/customer/recordings")
def customer_recordings():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id,filename,mime,size_bytes,created_at FROM ava_recordings WHERE customer_id=%s ORDER BY created_at DESC LIMIT 50",
            (session["customer_id"],))
        return jsonify([dict(r) for r in cur.fetchall()])

@app.route("/customer/recording/<rid>")
def customer_recording_file(rid):
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT filename,mime FROM ava_recordings WHERE id=%s AND customer_id=%s", (rid, session["customer_id"]))
        r = cur.fetchone()
    if not r: return jsonify({"error": "Not found"}), 404
    return send_file(os.path.join(RECORDINGS_DIR, r["filename"]), mimetype=r["mime"] or "video/webm")

# ── Privacy & disclosure report ──────────────────────────────────────────────
@app.route("/customer/people")
def customer_people():
    """Registered users this twin can be told to restrict, and Petar can set rules against."""
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id, name, last_seen, visit_count FROM ava_users ORDER BY last_seen DESC NULLS LAST LIMIT 200")
        return jsonify([dict(r) for r in cur.fetchall()])

@app.route("/customer/knowledge")
def customer_knowledge():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""SELECT id, chunk, category, ego_layer, visibility, allow_users, deny_users, created_at
            FROM ava_knowledge_base WHERE customer_id=%s ORDER BY created_at DESC""", (session["customer_id"],))
        return jsonify([dict(r) for r in cur.fetchall()])

@app.route("/customer/knowledge/<int:kid>/policy", methods=["POST"])
def set_knowledge_policy(kid):
    """Petar controls who may hear this memory. visibility: public|restricted. allow/deny: lists of user_id."""
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    vis = data.get("visibility", "public")
    if vis not in ("public", "restricted"): vis = "public"
    allow = json.dumps(data.get("allow_users", []))
    deny = json.dumps(data.get("deny_users", []))
    db = get_db()
    with db.cursor() as cur:
        cur.execute("UPDATE ava_knowledge_base SET visibility=%s, allow_users=%s::jsonb, deny_users=%s::jsonb WHERE id=%s AND customer_id=%s",
            (vis, allow, deny, kid, session["customer_id"]))
    db.commit()
    return jsonify({"ok": True})

@app.route("/customer/knowledge/<int:kid>", methods=["DELETE"])
def delete_knowledge(kid):
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM ava_knowledge_base WHERE id=%s AND customer_id=%s", (kid, session["customer_id"]))
    db.commit()
    return jsonify({"ok": True})

@app.route("/customer/me")
def customer_me():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db(); c = get_customer(db, session["customer_id"])
    return jsonify({"name": c["name"], "email": c["email"], "persona": c["persona_summary"] or ""})

# ════════════════════════════════ ADMIN PANEL ════════════════════════════════

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}; db = get_db()
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_admins WHERE email=%s AND password_hash=%s",
                        (data.get("email"), hash_pw(data.get("password", ""))))
            a = cur.fetchone()
        if a: session["admin_id"] = a["id"]; return jsonify({"ok": True})
        return jsonify({"error": "Invalid credentials"}), 401
    return render_template("admin/login.html")

@app.route("/admin/logout")
def admin_logout(): session.pop("admin_id", None); return redirect(url_for("admin_login"))

@app.route("/admin/")
def admin_home():
    if "admin_id" not in session: return redirect(url_for("admin_login"))
    return render_template("admin/index.html")

@app.route("/admin/customers", methods=["GET", "POST"])
def admin_customers():
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    if request.method == "POST":
        data = request.get_json(silent=True) or {}; cid = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute("INSERT INTO ava_customers (id,name,email,password_hash,voice_id,created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (cid, data["name"], data["email"], hash_pw(data["password"]),
                 (data.get("voice_id") or ELEVENLABS_VOICE_ID or "").strip(), datetime.utcnow().isoformat()))
        db.commit(); return jsonify({"ok": True, "id": cid})
    with db.cursor() as cur:
        cur.execute("SELECT id,name,email,voice_id,active,created_at FROM ava_customers ORDER BY created_at DESC")
        return jsonify([dict(c) for c in cur.fetchall()])

@app.route("/admin/customer/<cid>/update", methods=["POST"])
def admin_update_customer(cid):
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}; db = get_db()
    with db.cursor() as cur:
        if "voice_id" in data and data["voice_id"] is not None:
            cur.execute("UPDATE ava_customers SET voice_id=%s WHERE id=%s", (data["voice_id"].strip(), cid))
    db.commit(); return jsonify({"ok": True})

@app.route("/admin/toggle_customer/<cid>", methods=["POST"])
def admin_toggle_customer(cid):
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur: cur.execute("UPDATE ava_customers SET active=1-active WHERE id=%s", (cid,))
    db.commit(); return jsonify({"ok": True})

@app.route("/admin/stats")
def admin_stats():
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db(); stats = {}
    with db.cursor() as cur:
        for t, k in [("ava_customers","total_customers"),("ava_users","total_users"),("ava_sessions","total_sessions"),
                     ("ava_messages","total_messages"),("ava_knowledge_base","total_knowledge_chunks"),
                     ("ava_recordings","total_recordings")]:
            cur.execute(f"SELECT COUNT(*) FROM {t}"); stats[k] = cur.fetchone()["count"]
    return jsonify(stats)

# ── misc ─────────────────────────────────────────────────────────────────────
@app.route("/avatar/thumbnail")
def avatar_thumbnail():
    local = os.path.join(os.path.dirname(__file__), "static", "petar.png")
    if os.path.exists(local): return send_file(local, mimetype="image/png")
    return jsonify({"error": "No image"}), 404

@app.route("/healthz")
def healthz(): return jsonify({"ok": True})

@app.route("/healthz/ai")
def healthz_ai():
    """Hit this URL to see exactly whether the brain and embeddings work and why not."""
    sample = gemini_generate("Reply with the single word: OK")
    e = embed("hello world")
    return jsonify({
        "sdk_loaded": _genai_client is not None,
        "gemini_model": GEMINI_MODEL, "gemini_key_set": bool(GEMINI_API_KEY),
        "gemini_reply": sample, "gemini_error": _LAST_GEMINI_ERROR,
        "embed_model": GEMINI_EMBED_MODEL, "embed_dim_returned": (len(e) if e else 0),
        "elevenlabs_key_set": bool(ELEVENLABS_API_KEY), "elevenlabs_voice_id_set": bool(ELEVENLABS_VOICE_ID),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
