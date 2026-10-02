import os, datetime, json, uuid
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

# ---- Config (read from environment, never hard-code secrets) ----
DATABASE_URL = os.environ["DATABASE_URL"]
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = "HS256"
TOKEN_HOURS = 24

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ---- Database model ----
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

# ---- Request/response shapes ----
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

# ---- App ----
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

# ---- AI: script generation (Claude API) ----
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
    msg = client.messages.create(model="claude-sonnet-5", max_tokens=2000,
        messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except Exception:
        raise HTTPException(502, "The AI did not return valid JSON. Please try again.")
    return data

# ---- AI: real voice-over (free, no API key — Microsoft Edge TTS) ----
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

# ---- Same-origin test page for the two AI endpoints above ----
@app.get("/ai-test", response_class=HTMLResponse)
def ai_test_page():
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><title>AI test</title>
<style>body{font-family:system-ui;max-width:480px;margin:30px auto;padding:0 16px}
label{display:block;font-size:13px;color:#666;margin:10px 0 4px}
input,select,textarea{width:100%;box-sizing:border-box;padding:8px;border:1px solid #ccc;border-radius:4px}
button{margin-top:12px;padding:9px 14px;border:0;border-radius:4px;background:#f2a53c;font-weight:700;cursor:pointer}
pre{white-space:pre-wrap;background:#f4f4f4;padding:10px;border-radius:4px;font-size:13px}</style></head>
<body><h2>AI test page</h2>
<label>Access token (from /app, after logging in)</label><textarea id="tok" rows="2"></textarea>
<label>Topic</label><input id="topic" value="The history of the pyramids">
<label>Duration (seconds)</label><input id="dur" type="number" value="90">
<label>Language</label><select id="lang"><option>English</option><option>French</option><option>Moroccan Darija</option><option>Arabic</option><option>Spanish</option></select>
<button id="go1">Generate script</button>
<pre id="out1"></pre>
<div id="scenes"></div>
<script>
const $=id=>document.getElementById(id);
$("go1").onclick=async()=>{
 $("out1").textContent="Working…";
 const r=await fetch("/ai/script",{method:"POST",headers:{"Content-Type":"application/json","Authorization":"Bearer "+$("tok").value.trim()},
  body:JSON.stringify({topic:$("topic").value,duration_seconds:+$("dur").value,language:$("lang").value})});
 const d=await r.json();
 if(!r.ok){$("out1").textContent="Error: "+JSON.stringify(d);return}
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
};
</script></body></html>"""

# ---- Admin-only endpoints ----
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

# ---- One-time bootstrap: promote the very first admin ----
# Protected by ADMIN_BOOTSTRAP_KEY. Remove that env var from Render once used.
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
