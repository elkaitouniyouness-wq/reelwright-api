import os, datetime, json, uuid, subprocess, tempfile, textwrap
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, EmailStr
import bcrypt
from jose import jwt, JWTError
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from anthropic import Anthropic
import edge_tts
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
VOICE_MAP = {
 ("English","Female"):"en-US-AriaNeural", ("English","Male"):"en-US-GuyNeural",
 ("French","Female"):"fr-FR-DeniseNeural", ("French","Male"):"fr-FR-HenriNeural",
 ("Arabic","Female"):"ar-SA-ZariyahNeural", ("Arabic","Male"):"ar-SA-HamedNeural",
 ("Moroccan Darija","Female"):"ar-MA-MounaNeural", ("Moroccan Darija","Male"):"ar-MA-JamalNeural",
 ("Spanish","Female"):"es-ES-ElviraNeural", ("Spanish","Male"):"es-ES-AlvaroNeural",
 ("German","Female"):"de-DE-KatjaNeural", ("German","Male"):"de-DE-ConradNeural",
 ("Italian","Female"):"it-IT-ElsaNeural", ("Italian","Male"):"it-IT-DiegoNeural",
 ("Portuguese","Female"):"pt-PT-RaquelNeural", ("Portuguese","Male"):"pt-PT-DuarteNeural",
 ("Turkish","Female"):"tr-TR-EmelNeural", ("Turkish","Male"):"tr-TR-AhmetNeural",
}

DATABASE_URL = os.environ["DATABASE_URL"]
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = "HS256"
TOKEN_HOURS = 24

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user")
    plan = Column(String, default="Free")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SignupIn(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    model_config = {"from_attributes": True}
    id: int
    name: str
    email: str
    role: str
    plan: str

class BootstrapIn(BaseModel):
    email: EmailStr
    key: str

app = FastAPI(title="Reelwright API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_token(user: User) -> str:
    payload = {
        "sub": str(user.id),
        "role": user.role,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def current_user(authorization: str = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")
    user = db.get(User, int(data["sub"]))
    if not user or not user.is_active:
        raise HTTPException(401, "User not found or disabled")
    return user

def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin access required")
    return user

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/app")
def serve_app():
    return FileResponse("app.html")

@app.post("/auth/signup", response_model=UserOut)
def signup(body: SignupIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "An account with this email already exists")
    hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    user = User(name=body.name, email=body.email, password_hash=hashed)
    db.add(user); db.commit(); db.refresh(user)
    return user

@app.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not bcrypt.checkpw(body.password.encode(), user.password_hash.encode()):
        raise HTTPException(401, "Incorrect email or password")
    if not user.is_active:
        raise HTTPException(403, "This account has been suspended")
    return {"access_token": make_token(user), "token_type": "bearer",
            "user": UserOut.model_validate(user)}

@app.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user

class ScriptIn(BaseModel):
    topic: str
    duration_seconds: int = 180
    language: str = "English"

@app.post("/ai/script")
def ai_script(body: ScriptIn, user: User = Depends(current_user)):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY is not configured on the server yet")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    n_scenes = max(3, min(8, round(body.duration_seconds / 25)))
    prompt = (f"Write a {body.language} narration script for a {body.duration_seconds}-second video "
              f"about: {body.topic}. Split it into exactly {n_scenes} scenes. "
              f'Reply with ONLY valid JSON, no other text, no markdown fences, in this exact shape: '
              f'{{"scenes":[{{"title":"...","narration":"...","seconds":N}}]}}. '
              f"Make each scene's seconds roughly proportional to its narration length, summing to about {body.duration_seconds}.")
    msg = client.messages.create(model="claude-sonnet-5", max_tokens=4096,
        messages=[{"role": "user", "content": prompt}])
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text" and getattr(b, "text", None)), None)
    if not text:
        raise HTTPException(502, "The AI response had no text content. Please try again.")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except Exception:
        raise HTTPException(502, "The AI did not return valid JSON. Please try again.")
    return data

class VoiceIn(BaseModel):
    text: str
    language: str = "English"
    gender: str = "Female"

@app.post("/ai/voice")
async def ai_voice(body: VoiceIn, user: User = Depends(current_user)):
    voice = VOICE_MAP.get((body.language, body.gender), "en-US-AriaNeural")
    out_path = f"/tmp/{uuid.uuid4()}.mp3"
    communicate = edge_tts.Communicate(body.text, voice)
    await communicate.save(out_path)
    return FileResponse(out_path, media_type="audio/mpeg", filename="voiceover.mp3")

def _load_font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def make_slide(path, title, narration, w=1280, h=720):
    img = Image.new("RGB", (w, h), (46, 33, 54))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, 10], fill=(242, 165, 60))
    font_title = _load_font(56, bold=True)
    font_body = _load_font(34, bold=False)
    draw.text((60, 70), title, font=font_title, fill=(242, 165, 60))
    wrapped = textwrap.fill(narration, width=44)
    draw.multiline_text((60, 180), wrapped, font=font_body, fill=(244, 236, 232), spacing=14)
    img.save(path)

class VideoScene(BaseModel):
    title: str = ""
    narration: str = ""
    seconds: int = 0

class VideoIn(BaseModel):
    scenes: list[VideoScene]
    language: str = "English"
    gender: str = "Female"

@app.post("/ai/video")
async def ai_video(body: VideoIn, user: User = Depends(current_user)):
    if not body.scenes:
        raise HTTPException(400, "No scenes provided")
    voice = VOICE_MAP.get((body.language, body.gender), "en-US-AriaNeural")
    workdir = tempfile.mkdtemp()
    clip_paths = []
    for i, sc in enumerate(body.scenes):
        img_path = f"{workdir}/slide_{i}.png"
        audio_path = f"{workdir}/voice_{i}.mp3"
        clip_path = f"{workdir}/clip_{i}.mp4"
        make_slide(img_path, sc.title, sc.narration)
        await edge_tts.Communicate(sc.narration, voice).save(audio_path)
        proc = subprocess.run([FFMPEG, "-y", "-loop", "1", "-i", img_path, "-i", audio_path,
            "-c:v", "libx264", "-tune", "stillimage", "-r", "25", "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", "-shortest", clip_path], capture_output=True)
        if proc.returncode != 0:
            raise HTTPException(500, f"ffmpeg failed on scene {i}: {proc.stderr.decode(errors='ignore')[:400]}")
        clip_paths.append(clip_path)
    concat_path = f"{workdir}/concat.txt"
    with open(concat_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    final_path = f"{workdir}/final.mp4"
    proc = subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final_path], capture_output=True)
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg concat failed: {proc.stderr.decode(errors='ignore')[:400]}")
    return FileResponse(final_path, media_type="video/mp4", filename="reelwright-video.mp4")

@app.get("/ai-test", response_class=HTMLResponse)
def ai_test_page():
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI test</title>
<style>body{font-family:system-ui;max-width:480px;margin:30px auto;padding:0 16px}
label{display:block;font-size:13px;color:#666;margin:10px 0 4px}
input,select,textarea{width:100%;box-sizing:border-box;padding:8px;border:1px solid #ccc;border-radius:4px}
button{margin-top:12px;padding:9px 14px;border:0;border-radius:4px;background:#f2a53c;font-weight:700;cursor:pointer}
pre{white-space:pre-wrap;background:#f4f4f4;padding:10px;border-radius:4px;font-size:13px}
video{width:100%;margin-top:10px}</style></head>
<body><h2>AI test page</h2>
<label>Access token (from /app, after logging in)</label><textarea id="tok" rows="2"></textarea>
<label>Topic</label><input id="topic" value="The history of the pyramids">
<label>Duration (seconds)</label><input id="dur" type="number" value="45">
<label>Language</label><select id="lang"><option>English</option><option>French</option><option>Moroccan Darija</option><option>Arabic</option><option>Spanish</option></select>
<button id="go1">Generate script</button>
<pre id="out1"></pre>
<div id="scenes"></div>
<button id="govid" style="display:none;background:#5bc48a">Generate full video</button>
<div id="vidout"></div>
<script>
const $=id=>document.getElementById(id);
let lastScript=null;
$("go1").onclick=async()=>{
 $("out1").textContent="Working…";
 const r=await fetch("/ai/script",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+$("tok").value.trim()},
  body:JSON.stringify({topic:$("topic").value,duration_seconds:+$("dur").value,language:$("lang").value})});
 const d=await r.json();
 if(!r.ok){$("out1").textContent="Error: "+JSON.stringify(d);return}
 lastScript=d;
 $("out1").textContent=JSON.stringify(d,null,2);
 $("scenes").innerHTML=d.scenes.map((s,i)=>`<div style="margin-top:10px;padding:10px;border:1px solid #ddd;border-radius:4px">
  <b>${s.title}</b> (${s.seconds}s)<p>${s.narration}</p>
  <button data-i="${i}" class="voiceBtn">Generate voice</button><div id="a${i}"></div></div>`).join("");
 document.querySelectorAll(".voiceBtn").forEach(b=>b.onclick=async()=>{
  const i=b.dataset.i,sc=d.scenes[i];b.textContent="Working…";
  const r2=await fetch("/ai/voice",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+$("tok").value.trim()},
   body:JSON.stringify({text:sc.narration,language:$("lang").value,gender:"Female"})});
  if(!r2.ok){$("a"+i).textContent="Error generating voice";return}
  const blob=await r2.blob();const url=URL.createObjectURL(blob);
  $("a"+i).innerHTML=`<audio controls src="${url}"></audio>`;b.textContent="Generate voice"});
 $("govid").style.display="block";
};
$("govid").onclick=async()=>{
 if(!lastScript)return;
 $("vidout").textContent="Rendering video… this can take a minute or two.";
 const r=await fetch("/ai/video",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+$("tok").value.trim()},
  body:JSON.stringify({scenes:lastScript.scenes,language:$("lang").value,gender:"Female"})});
 if(!r.ok){const d=await r.json().catch(()=>({}));$("vidout").textContent="Error: "+JSON.stringify(d);return}
 const blob=await r.blob();const url=URL.createObjectURL(blob);
 $("vidout").innerHTML=`<video controls src="${url}"></video><br><a href="${url}" download="reelwright-video.mp4">Download MP4</a>`;
};
</script></body></html>"""

@app.get("/admin/users", response_model=list[UserOut])
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at.desc()).all()

@app.post("/admin/users/{user_id}/suspend")
def suspend_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    u.is_active = False
    db.commit()
    return {"status": "suspended", "user_id": user_id}

@app.post("/admin/users/{user_id}/make-admin")
def make_admin(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    u.role = "admin"
    db.commit()
    return {"status": "promoted", "user_id": user_id}

@app.post("/admin/bootstrap", response_model=UserOut)
def bootstrap_admin(body: BootstrapIn, db: Session = Depends(get_db)):
    expected = os.environ.get("ADMIN_BOOTSTRAP_KEY")
    if not expected or body.key != expected:
        raise HTTPException(403, "Invalid bootstrap key")
    user = db.query(User).filter(User.email == body.email).first()
    if not user:
        raise HTTPException(404, "No account with this email yet — sign up first")
    user.role = "admin"
    db.commit(); db.refresh(user)
    return user
