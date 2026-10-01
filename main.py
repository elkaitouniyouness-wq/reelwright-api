import os, datetime
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import jwt, JWTError
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = os.environ["DATABASE_URL"]
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = "HS256"
TOKEN_HOURS = 24

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

@app.post("/auth/signup", response_model=UserOut)
def signup(body: SignupIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "An account with this email already exists")
    user = User(name=body.name, email=body.email, password_hash=pwd.hash(body.password))
    db.add(user); db.commit(); db.refresh(user)
    return user

@app.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not pwd.verify(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password")
    if not user.is_active:
        raise HTTPException(403, "This account has been suspended")
    return {"access_token": make_token(user), "token_type": "bearer",
            "user": UserOut.model_validate(user)}

@app.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user

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

class BootstrapIn(BaseModel):
    email: EmailStr
    key: str

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
from fastapi.responses import HTMLResponse

@app.get("/tools", response_class=HTMLResponse)
def tools_page():
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Reelwright — Account tools</title>
<style>
body{margin:0;background:#171019;color:#f4ece8;font:15px/1.5 system-ui,sans-serif;padding:30px 16px}
h1{font-size:24px;margin:0 0 4px}.mu{color:#a9939f}.wrap{max-width:480px;margin:0 auto}
.card{background:#211823;border:1px solid #40323f;border-radius:6px;padding:18px;margin-top:18px}
h2{font-size:17px;margin:0 0 10px}
label{display:block;font-size:12.5px;color:#a9939f;margin:10px 0 4px}
input{width:100%;box-sizing:border-box;background:#171019;border:1px solid #40323f;border-radius:4px;padding:9px 10px;color:#f4ece8;font:inherit}
button{margin-top:14px;width:100%;background:#f2a53c;border:0;border-radius:4px;padding:10px;color:#231405;font-weight:700;font:inherit;cursor:pointer}
.out{margin-top:12px;font-size:13px;white-space:pre-wrap;word-break:break-word;padding:10px;border-radius:4px}
.ok{background:#12321f;color:#8be0ab}.err{background:#3a141c;color:#ff9aac}
</style></head><body><div class="wrap">
<h1>Reelwright — Account tools</h1>
<p class="mu">Served directly from your API, so there are no cross-origin issues.</p>
<div class="card"><h2>1. Create your account</h2>
<label>Name</label><input id="su_name">
<label>Email</label><input id="su_email" type="email">
<label>Password</label><input id="su_pass" type="password">
<button id="su_go">Sign up</button><div id="su_out"></div></div>
<div class="card"><h2>2. Log in</h2>
<label>Email</label><input id="li_email" type="email">
<label>Password</label><input id="li_pass" type="password">
<button id="li_go">Log in</button><div id="li_out"></div></div>
<div class="card"><h2>3. Make this account admin (one-time)</h2>
<label>Email</label><input id="bs_email" type="email">
<label>Bootstrap key</label><input id="bs_key" type="password">
<button id="bs_go">Make admin</button><div id="bs_out"></div></div>
<div class="card"><h2>4. Check who you are</h2>
<label>Access token</label><input id="me_tok">
<button id="me_go">Check /me</button><div id="me_out"></div></div>
</div>
<script>
const $=id=>document.getElementById(id);
async function call(el,method,path,body,tok){
 el.className="out";el.textContent="Working…";
 try{
  const r=await fetch(path,{method,headers:{"Content-Type":"application/json",...(tok?{Authorization:"Bearer "+tok}:{})},body:body?JSON.stringify(body):undefined});
  const d=await r.json();
  el.className="out "+(r.ok?"ok":"err");el.textContent=JSON.stringify(d,null,2);
  return {ok:r.ok,data:d};
 }catch(e){el.className="out err";el.textContent="Network error: "+e.message;return{ok:false}}
}
$("su_go").onclick=()=>call($("su_out"),"POST","/auth/signup",{name:$("su_name").value,email:$("su_email").value,password:$("su_pass").value});
$("li_go").onclick=async()=>{const r=await call($("li_out"),"POST","/auth/login",{email:$("li_email").value,password:$("li_pass").value});
 if(r.ok&&r.data.access_token){$("me_tok").value=r.data.access_token}};
$("bs_go").onclick=()=>call($("bs_out"),"POST","/admin/bootstrap",{email:$("bs_email").value,key:$("bs_key").value});
$("me_go").onclick=()=>call($("me_out"),"GET","/me",null,$("me_tok").value);
</script></body></html>"""
