import os, uuid, base64, hashlib, json, threading, time
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, Response, g, session, redirect, url_for
from flask_cors import CORS
import requests
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "trione-secret-change-in-prod")
CORS(app)

DATABASE_URL    = os.getenv("DATABASE_URL")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
LIVEAVATAR_API_KEY  = os.getenv("LIVEAVATAR_API_KEY")
LIVEAVATAR_AVATAR_ID = os.getenv("LIVEAVATAR_AVATAR_ID")

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
            password_hash TEXT NOT NULL, avatar_id TEXT, elevenlabs_agent_id TEXT,
            elevenlabs_secret_id TEXT, persona_summary TEXT, ego_model TEXT,
            created_at TEXT, active INTEGER DEFAULT 1)""",
        """CREATE TABLE IF NOT EXISTS ava_users (
            id TEXT PRIMARY KEY, face_encoding TEXT, name TEXT DEFAULT 'Guest',
            first_seen TEXT, last_seen TEXT, visit_count INTEGER DEFAULT 1)""",
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

init_db()

# ── Seed admin ────────────────────────────────────────────────────────────────

def seed_admin():
    with psycopg2.connect(DATABASE_URL) as db:
        with db.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM ava_admins")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO ava_admins (id,email,password_hash,created_at) VALUES (%s,%s,%s,%s)",
                    (str(uuid.uuid4()), "admin@example.com",
                     hashlib.sha256("password".encode()).hexdigest(),
                     datetime.utcnow().isoformat())
                )
        db.commit()
        print("Default admin: admin@example.com / password")

seed_admin()

# ── Seed customer from env ────────────────────────────────────────────────────

def seed_customer():
    avatar_id = LIVEAVATAR_AVATAR_ID
    agent_id  = ELEVENLABS_AGENT_ID
    if not avatar_id or not agent_id:
        return
    with psycopg2.connect(DATABASE_URL) as db:
        with db.cursor() as cur:
            cur.execute("UPDATE ava_customers SET avatar_id=%s, elevenlabs_agent_id=%s WHERE avatar_id IS NULL OR avatar_id=''",
                        (avatar_id, agent_id))
        db.commit()

seed_customer()

# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_customer(db, cid):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM ava_customers WHERE id=%s", (cid,))
        return cur.fetchone()

def get_first_customer(db):
    with db.cursor() as cur:
        cur.execute("SELECT * FROM ava_customers WHERE active=1 LIMIT 1")
        return cur.fetchone()

def liveavatar_session_token(avatar_id, elevenlabs_secret_id, elevenlabs_agent_id):
    r = requests.post(
        "https://api.liveavatar.com/v1/sessions/token",
        headers={"X-API-KEY": LIVEAVATAR_API_KEY, "Content-Type": "application/json"},
        json={
            "avatar_id": avatar_id,
            "mode": "LITE",
            "elevenlabs_agent_config": {
                "secret_id": elevenlabs_secret_id.strip(),
                "agent_id": elevenlabs_agent_id.strip()
            }
        }
    )
    if r.status_code not in [200, 201]:
        return None, r.text
    data = r.json()
    token = data.get("data", {}).get("session_token") or data.get("session_token")
    return token, None

def gemini(prompt, system=None):
    if not GEMINI_API_KEY: return ""
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

def parse_json(text):
    text = text.strip().strip("```json").strip("```").strip()
    try: return json.loads(text)
    except: return None

# ── EGO Engine ────────────────────────────────────────────────────────────────

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
    with psycopg2.connect(DATABASE_URL) as db:
        with db.cursor() as cur:
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
            cur.execute("INSERT INTO ava_ego_revisions (customer_id, revision, trigger_text, created_at) VALUES (%s,%s,%s,%s)",
                        (customer_id, ego_json, trigger_text[:500], datetime.utcnow().isoformat()))
        db.commit()

def get_ego_context(db, customer_id):
    with db.cursor() as cur:
        cur.execute("SELECT ego_model, persona_summary FROM ava_customers WHERE id=%s", (customer_id,))
        c = cur.fetchone()
        cur.execute("SELECT chunk, category FROM ava_knowledge_base WHERE customer_id=%s ORDER BY created_at DESC LIMIT 40", (customer_id,))
        chunks = cur.fetchall()

    ego = None
    if c and c["ego_model"]:
        try: ego = json.loads(c["ego_model"])
        except: pass

    parts = []
    if ego:
        if ego.get("summary"): parts.append(f"=== PERSONA ===\n{ego['summary']}")
        if ego.get("beliefs"): parts.append("=== BELIEFS ===\n" + "\n".join(f"• {b}" for b in ego["beliefs"][:10]))
        if ego.get("values"): parts.append("=== VALUES ===\n" + "\n".join(f"• {v}" for v in ego["values"][:8]))
        if ego.get("reasoning_patterns"): parts.append("=== HOW THEY THINK ===\n" + "\n".join(f"• {r}" for r in ego["reasoning_patterns"][:6]))
        if ego.get("relationships"):
            parts.append("=== RELATIONSHIPS ===\n" + "\n".join(f"• {p}: {d}" for p,d in list(ego["relationships"].items())[:10]))
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
    except Exception as e: print(f"Face encode error: {e}"); return None

def find_face(db, enc, tolerance=0.5):
    try:
        import face_recognition
        with db.cursor() as cur:
            cur.execute("SELECT * FROM ava_users WHERE face_encoding IS NOT NULL")
            for u in cur.fetchall():
                stored = np.array(json.loads(u["face_encoding"]))
                if face_recognition.face_distance([stored], enc)[0] < tolerance:
                    return dict(u)
    except Exception as e: print(f"Face match error: {e}")
    return None

# ════════════════════════════════════════════════════════════════════════════
# USER PORTAL
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def user_home():
    return render_template("user/index.html")

@app.route("/u/identify", methods=["GET", "POST"])
def user_identify():
    data = request.get_json(silent=True) or {}
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
                user_id = match["id"]; name = match["name"]; returning = True
                with db.cursor() as cur:
                    cur.execute("UPDATE ava_users SET last_seen=%s, visit_count=visit_count+1 WHERE id=%s",
                               (datetime.utcnow().isoformat(), user_id))
            else:
                with db.cursor() as cur:
                    cur.execute("INSERT INTO ava_users (id, face_encoding, name, first_seen, last_seen) VALUES (%s,%s,%s,%s,%s)",
                               (user_id, json.dumps(enc.tolist()), "Guest",
                                datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
            db.commit()

    with db.cursor() as cur:
        cur.execute("SELECT speaker, text FROM ava_messages WHERE user_id=%s ORDER BY timestamp DESC LIMIT 20", (user_id,))
        history = cur.fetchall()

    return jsonify({
        "user_id": user_id, "name": name, "returning": returning,
        "history": [{"speaker": r["speaker"], "text": r["text"]} for r in reversed(history)]
    })

@app.route("/u/session", methods=["GET", "POST"])
def user_session():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", str(uuid.uuid4()))
    db = get_db()

    customer = get_first_customer(db)
    if not customer:
        return jsonify({"error": "No active avatar found"}), 404

    rag_context = get_ego_context(db, customer["id"])
    session_id = str(uuid.uuid4())
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_sessions (id, customer_id, user_id, session_type, started_at) VALUES (%s,%s,%s,%s,%s)",
                   (session_id, customer["id"], user_id, "user_chat", datetime.utcnow().isoformat()))
    db.commit()

    avatar_id  = (customer["avatar_id"] or "").strip() or LIVEAVATAR_AVATAR_ID
    secret_id  = (customer["elevenlabs_secret_id"] or "").strip()
    agent_id   = (customer["elevenlabs_agent_id"] or "").strip() or ELEVENLABS_AGENT_ID

    if not avatar_id or not secret_id or not agent_id:
        return jsonify({"error": "Avatar not fully configured. Set avatar_id, elevenlabs_secret_id and elevenlabs_agent_id in admin panel."}), 500

    token, err = liveavatar_session_token(avatar_id, secret_id, agent_id)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({
        "session_token": token,
        "session_id": session_id,
        "customer_id": customer["id"],
        "customer_name": customer["name"],
        "rag_context": rag_context,
        "persona": customer["persona_summary"] or ""
    })

@app.route("/u/message", methods=["GET", "POST"])
def user_message():
    data = request.get_json(silent=True) or {}
    db = get_db()
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_messages (session_id, customer_id, user_id, speaker, text, timestamp) VALUES (%s,%s,%s,%s,%s,%s)",
                   (data.get("session_id"), data.get("customer_id"), data.get("user_id"),
                    data.get("speaker"), data.get("text"), datetime.utcnow().isoformat()))
    db.commit()
    return jsonify({"ok": True})

# ════════════════════════════════════════════════════════════════════════════
# CUSTOMER PORTAL
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

@app.route("/customer/session", methods=["GET", "POST"])
def customer_session():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    customer = get_customer(db, session["customer_id"])
    if not customer: return jsonify({"error": "Not found"}), 404

    session_id = str(uuid.uuid4())
    with db.cursor() as cur:
        cur.execute("INSERT INTO ava_sessions (id, customer_id, session_type, started_at) VALUES (%s,%s,%s,%s)",
                   (session_id, customer["id"], "customer_training", datetime.utcnow().isoformat()))
    db.commit()

    avatar_id = (customer["avatar_id"] or "").strip() or LIVEAVATAR_AVATAR_ID
    secret_id = (customer["elevenlabs_secret_id"] or "").strip()
    agent_id  = (customer["elevenlabs_agent_id"] or "").strip() or ELEVENLABS_AGENT_ID

    token = None
    if avatar_id and secret_id and agent_id:
        token, err = liveavatar_session_token(avatar_id, secret_id, agent_id)
        if err:
            return jsonify({"error": err}), 500

    with db.cursor() as cur:
        cur.execute("SELECT COUNT(*) as cnt FROM ava_knowledge_base WHERE customer_id=%s", (customer["id"],))
        knowledge_count = cur.fetchone()["cnt"]

    ego = None
    if customer["ego_model"]:
        try: ego = json.loads(customer["ego_model"])
        except: pass

    return jsonify({
        "session_token": token,
        "session_id": session_id,
        "customer_id": customer["id"],
        "knowledge_count": knowledge_count,
        "persona": customer["persona_summary"] or "",
        "ego_model": ego
    })

@app.route("/customer/train", methods=["GET", "POST"])
def customer_train():
    if "customer_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
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
                cur.execute("INSERT INTO ava_knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                           (customer_id, chunk["chunk"], session_id, chunk.get("category", "fact"), "raw", datetime.utcnow().isoformat()))
                chunks_added += 1
        for belief in ego_data.get("beliefs", []):
            cur.execute("INSERT INTO ava_knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                       (customer_id, belief, session_id, "belief", "ego", datetime.utcnow().isoformat()))
        for value in ego_data.get("values", []):
            cur.execute("INSERT INTO ava_knowledge_base (customer_id, chunk, source_session_id, category, ego_layer, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                       (customer_id, value, session_id, "value", "ego", datetime.utcnow().isoformat()))
    db.commit()

    threading.Thread(target=update_ego_model, args=(customer_id, ego_data, text), daemon=True).start()

    return jsonify({"ok": True, "chunks_added": chunks_added,
                    "beliefs_extracted": len(ego_data.get("beliefs", [])),
                    "values_extracted": len(ego_data.get("values", []))})

@app.route("/customer/ego", methods=["GET", "POST"])
def customer_ego():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    c = get_customer(db, session["customer_id"])
    ego = None
    if c["ego_model"]:
        try: ego = json.loads(c["ego_model"])
        except: pass
    return jsonify({"ego": ego, "summary": c["persona_summary"] or ""})

@app.route("/customer/knowledge", methods=["GET", "POST"])
def customer_knowledge():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id, chunk, category, ego_layer, created_at FROM ava_knowledge_base WHERE customer_id=%s ORDER BY created_at DESC",
                   (session["customer_id"],))
        chunks = cur.fetchall()
    return jsonify([dict(c) for c in chunks])

@app.route("/customer/knowledge/<int:kid>", methods=["GET", "POST", "DELETE"])
def delete_knowledge(kid):
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM ava_knowledge_base WHERE id=%s AND customer_id=%s", (kid, session["customer_id"]))
    db.commit()
    return jsonify({"ok": True})

@app.route("/customer/sessions", methods=["GET", "POST"])
def customer_sessions():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""SELECT s.id, s.session_type, s.started_at, COUNT(m.id) as message_count
                       FROM ava_sessions s LEFT JOIN ava_messages m ON s.id=m.session_id
                       WHERE s.customer_id=%s GROUP BY s.id, s.session_type, s.started_at
                       ORDER BY s.started_at DESC LIMIT 50""", (session["customer_id"],))
        rows = cur.fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/customer/me", methods=["GET", "POST"])
def customer_me():
    if "customer_id" not in session: return jsonify({"error": "Not logged in"}), 401
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
    if "admin_id" not in session: return redirect(url_for("admin_login"))
    return render_template("admin/index.html")

@app.route("/admin/customers", methods=["GET", "POST"])
def admin_customers():
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        db = get_db()
        cid = str(uuid.uuid4())
        with db.cursor() as cur:
            cur.execute(
                "INSERT INTO ava_customers (id, name, email, password_hash, avatar_id, elevenlabs_agent_id, elevenlabs_secret_id, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (cid, data["name"], data["email"], hash_pw(data["password"]),
                 (data.get("avatar_id") or LIVEAVATAR_AVATAR_ID or "").strip(),
                 (data.get("elevenlabs_agent_id") or ELEVENLABS_AGENT_ID or "").strip(),
                 (data.get("elevenlabs_secret_id") or "").strip(),
                 datetime.utcnow().isoformat())
            )
        db.commit()
        return jsonify({"ok": True, "id": cid})
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT id, name, email, avatar_id, elevenlabs_agent_id, elevenlabs_secret_id, active, created_at FROM ava_customers ORDER BY created_at DESC")
        rows = cur.fetchall()
    return jsonify([dict(c) for c in rows])

@app.route("/admin/customer/<cid>/update", methods=["GET", "POST"])
def admin_update_customer(cid):
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    db = get_db()
    with db.cursor() as cur:
        for field in ["elevenlabs_secret_id", "elevenlabs_agent_id", "avatar_id"]:
            if field in data and data[field]:
                cur.execute(f"UPDATE ava_customers SET {field}=%s WHERE id=%s", (data[field].strip(), cid))
    db.commit()
    return jsonify({"ok": True})

@app.route("/admin/toggle_customer/<cid>", methods=["GET", "POST"])
def admin_toggle_customer(cid):
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        cur.execute("UPDATE ava_customers SET active=1-active WHERE id=%s", (cid,))
    db.commit()
    return jsonify({"ok": True})

@app.route("/admin/stats", methods=["GET", "POST"])
def admin_stats():
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    with db.cursor() as cur:
        stats = {}
        for table, key in [("ava_customers","total_customers"),("ava_users","total_users"),
                           ("ava_sessions","total_sessions"),("ava_messages","total_messages"),
                           ("ava_knowledge_base","total_knowledge_chunks"),("ava_ego_revisions","total_ego_revisions")]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[key] = cur.fetchone()["count"]
    return jsonify(stats)

@app.route("/admin/customer/<cid>/ego", methods=["GET", "POST"])
def admin_customer_ego(cid):
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    db = get_db()
    c = get_customer(db, cid)
    if not c: return jsonify({"error": "Not found"}), 404
    ego = None
    if c["ego_model"]:
        try: ego = json.loads(c["ego_model"])
        except: pass
    return jsonify({"name": c["name"], "persona": c["persona_summary"], "ego": ego})

@app.route("/admin/register_elevenlabs_secret", methods=["GET", "POST"])
def register_elevenlabs_secret():
    if "admin_id" not in session: return jsonify({"error": "Not logged in"}), 401
    data = request.get_json(silent=True) or {}
    el_api_key = data.get("elevenlabs_api_key")
    if not el_api_key: return jsonify({"error": "Missing elevenlabs_api_key"}), 400
    r = requests.post(
        "https://api.liveavatar.com/v1/secrets",
        headers={"X-API-KEY": LIVEAVATAR_API_KEY, "Content-Type": "application/json"},
        json={"secret_type": "ELEVENLABS_API_KEY", "secret_value": el_api_key, "secret_name": data.get("name", "ElevenLabs Key")}
    )
    if r.status_code not in [200, 201]: return jsonify({"error": r.text}), 500
    return jsonify(r.json())

# ── Avatar thumbnail ──────────────────────────────────────────────────────────

@app.route("/avatar/thumbnail")
def avatar_thumbnail():
    db = get_db()
    customer = get_first_customer(db)
    avatar_id = ((customer["avatar_id"] or "").strip() if customer else "") or LIVEAVATAR_AVATAR_ID
    if avatar_id:
        try:
            r = requests.get(f"https://api.liveavatar.com/v2/avatars/{avatar_id}",
                             headers={"X-API-KEY": LIVEAVATAR_API_KEY})
            if r.status_code == 200:
                d = r.json().get("data", {})
                thumb = d.get("thumbnail_url") or d.get("preview_url")
                if thumb:
                    img = requests.get(thumb)
                    return Response(img.content, mimetype=img.headers.get("Content-Type", "image/jpeg"))
        except: pass
    local = os.path.join(os.path.dirname(__file__), "static", "petar.jpg")
    if os.path.exists(local): return send_file(local, mimetype="image/jpeg")
    return jsonify({"error": "No image"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
