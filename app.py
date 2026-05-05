from flask import Flask, render_template, request, jsonify
import requests
import os
import pprint

app = Flask(__name__)

LIVEAVATAR_API_KEY   = os.environ.get('LIVEAVATAR_API_KEY', '')
LIVEAVATAR_AVATAR_ID = os.environ.get('LIVEAVATAR_AVATAR_ID', '')
LIVEAVATAR_CONTEXT_ID= os.environ.get('LIVEAVATAR_CONTEXT_ID', '17a29e59-031a-490b-9c54-ff5acd170005')
ELEVENLABS_AGENT_ID  = os.environ.get('ELEVENLABS_AGENT_ID', '')
ELEVENLABS_API_KEY   = os.environ.get('ELEVENLABS_API_KEY', '')
# After running /api/register-elevenlabs once, paste the returned secret_id here
ELEVENLABS_SECRET_ID = os.environ.get('ELEVENLABS_SECRET_ID', '')
LIVEAVATAR_VOICE_ID  = os.environ.get('LIVEAVATAR_VOICE_ID', '51afbab6-7af4-473b-95fc-6ce26aac8bb1')

_active_session_id = None


def stop_session(session_id):
    if not session_id:
        return
    try:
        r = requests.post(
            'https://api.liveavatar.com/v1/sessions/stop',
            headers={'X-API-KEY': LIVEAVATAR_API_KEY, 'Content-Type': 'application/json'},
            json={'session_id': session_id},
            timeout=5,
        )
        print(f"[stop_session] {session_id} → {r.status_code}")
    except Exception as e:
        print(f"[stop_session] error: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/demo', methods=['POST'])
def demo():
    user_name = request.form.get('user_name', 'Guest')
    return render_template('demo.html',
        user_name=user_name,
        avatar_id=LIVEAVATAR_AVATAR_ID,
    )


@app.route('/api/register-elevenlabs', methods=['POST'])
def register_elevenlabs():
    """
    ONE-TIME SETUP: registers your ElevenLabs API key with LiveAvatar.
    Call this once from the browser or curl, then store the returned secret_id
    in your .env as ELEVENLABS_SECRET_ID.
    """
    if not ELEVENLABS_API_KEY:
        return jsonify({'error': 'ELEVENLABS_API_KEY not set'}), 500
    try:
        r = requests.post(
            'https://api.liveavatar.com/v1/secrets',
            headers={'X-Api-Key': LIVEAVATAR_API_KEY, 'Content-Type': 'application/json'},
            json={
                'secret_type': 'ELEVENLABS_API_KEY',
                'secret_value': ELEVENLABS_API_KEY,
                'secret_name': 'elevenlabs_key',
            },
            timeout=10,
        )
        data = r.json()
        print("\n=== register-elevenlabs response ===")
        pprint.pprint(data)
        print("====================================\n")
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/heygen-token', methods=['POST'])
def heygen_token():
    global _active_session_id

    if not LIVEAVATAR_API_KEY:
        return jsonify({'error': 'LIVEAVATAR_API_KEY not set'}), 500
    if not ELEVENLABS_SECRET_ID:
        return jsonify({'error': 'ELEVENLABS_SECRET_ID not set — call /api/register-elevenlabs first'}), 500
    if not ELEVENLABS_AGENT_ID:
        return jsonify({'error': 'ELEVENLABS_AGENT_ID not set'}), 500

    if _active_session_id:
        print(f"[token] Stopping previous session: {_active_session_id}")
        stop_session(_active_session_id)
        _active_session_id = None

    body = {
        'mode': 'LITE',
        'avatar_id': LIVEAVATAR_AVATAR_ID,
        'is_sandbox': False,
        'elevenlabs_agent_config': {
            'secret_id': ELEVENLABS_SECRET_ID,
            'agent_id': ELEVENLABS_AGENT_ID,
        },
        'avatar_persona': {
            'voice_id': LIVEAVATAR_VOICE_ID,
            'language': 'en',
        },
    }

    try:
        resp = requests.post(
            'https://api.liveavatar.com/v1/sessions/token',
            headers={'X-API-KEY': LIVEAVATAR_API_KEY, 'Content-Type': 'application/json'},
            json=body,
            timeout=10,
        )
        raw = resp.json()
        print("\n=== /v1/sessions/token response ===")
        pprint.pprint(raw)
        print("===================================\n")

        if resp.status_code != 200:
            return jsonify({'error': f'LiveAvatar {resp.status_code}', 'raw': raw}), 500

        payload = raw.get('data') or raw
        session_token = (payload.get('session_token') or payload.get('sessionToken') or payload.get('token'))
        session_id    = (payload.get('session_id')    or payload.get('session_Id')   or payload.get('sessionId'))

        if not session_token:
            return jsonify({'error': 'No token in response', 'raw': raw}), 500

        _active_session_id = session_id
        return jsonify({'token': session_token, 'session_id': session_id})

    except requests.RequestException as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop-session', methods=['POST'])
def stop_session_route():
    global _active_session_id
    sid = (request.json or {}).get('session_id')
    stop_session(sid or _active_session_id)
    _active_session_id = None
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
