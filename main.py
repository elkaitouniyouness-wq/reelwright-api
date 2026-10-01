import os, datetime
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
import bcrypt
from jose import jwt, JWTError
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

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

# ---- Admin-only endpoints ----
@app.get("/admin/users", response_model=list[UserOut])
def list_users(admin: User =
