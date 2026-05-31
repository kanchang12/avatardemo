from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import httpx
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
NGROK_URL = os.getenv("NGROK_URL")
AVATAR_IMAGE = os.getenv("AVATAR_IMAGE", "static/avatar.jpg")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.get("/get_signed_url")
async def get_signed_url():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.elevenlabs.io/v1/convai/conversation/get_signed_url?agent_id={ELEVENLABS_AGENT_ID}",
            headers={"xi-api-key": ELEVENLABS_API_KEY}
        )
        data = response.json()
        return {"signed_url": data.get("signed_url")}


@app.post("/lipsync")
async def lipsync(payload: dict):
    audio_b64 = payload.get("audio_b64")
    if not audio_b64:
        return {"error": "no audio"}

    audio_bytes = base64.b64decode(audio_b64)
    with open("/tmp/agent_audio.wav", "wb") as f:
        f.write(audio_bytes)

    musetalk_url = f"{NGROK_URL}/talker_response/"

    async with httpx.AsyncClient(timeout=60) as client:
        with open("/tmp/agent_audio.wav", "rb") as af, open(AVATAR_IMAGE, "rb") as imgf:
            response = await client.post(
                musetalk_url,
                files={
                    "source_image": ("avatar.jpg", imgf, "image/jpeg"),
                    "driven_audio": ("audio.wav", af, "audio/wav"),
                },
                data={"talker_method": "SadTalker"}
            )

    if response.status_code == 200:
        video_b64 = base64.b64encode(response.content).decode()
        return {"video_b64": video_b64}
    else:
        return {"error": response.text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
