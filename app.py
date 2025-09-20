# app.py
"""
FinanceAI — Streamlit single-file app (IBM Watson only)
- Profile (sidebar)
- Welcome quick-actions
- Chat UI
- Message classification
- LLM inference via IBM Watsonx Granite
- IBM Watson STT & TTS integration
- Persistence via pandas
"""

import streamlit as st
import os, time, json, io
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

# IBM SDKs
from ibm_watson import SpeechToTextV1, TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

# ---------------------------
# Config - Using Environment Variables
# ---------------------------
IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_STT_URL = os.getenv("IBM_STT_URL")
IBM_TTS_URL = os.getenv("IBM_TTS_URL")
IBM_WATSONX_URL = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")  # required for watsonx.ai

# Validate config
if not IBM_API_KEY:
    st.error("❌ IBM_API_KEY not set in environment.")
if not IBM_PROJECT_ID:
    st.error("❌ IBM_PROJECT_ID not set in environment.")

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
PROFILE_PATH = DATA_DIR / "profile.json"
MESSAGES_PATH = DATA_DIR / "messages.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Profile model
# ---------------------------
@dataclass
class Profile:
    user_type: str = ""
    age: int = 0
    monthly_income: int = 0
    financial_goals: List[str] = None
    risk_tolerance: str = ""
    has_emergency_fund: bool = False
    current_debt: int = 0
    profile_completed: bool = False
    def to_dict(self): return asdict(self)

# ---------------------------
# Persistence
# ---------------------------
def load_profile() -> Profile:
    if PROFILE_PATH.exists():
        return Profile(**json.loads(PROFILE_PATH.read_text()))
    return Profile()

def save_profile(profile: Profile):
    PROFILE_PATH.write_text(json.dumps(profile.to_dict(), indent=2))

def load_messages() -> pd.DataFrame:
    if MESSAGES_PATH.exists():
        return pd.read_csv(MESSAGES_PATH)
    return pd.DataFrame(columns=["id","timestamp","is_user","message","message_type"])

def save_message_row(row: Dict):
    df = load_messages()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(MESSAGES_PATH, index=False)

# ---------------------------
# IBM Watson setup
# ---------------------------
def init_stt():
    if not (IBM_API_KEY and IBM_STT_URL): return None
    auth = IAMAuthenticator(IBM_API_KEY)
    stt = SpeechToTextV1(authenticator=auth)
    stt.set_service_url(IBM_STT_URL)
    return stt

def init_tts():
    if not (IBM_API_KEY and IBM_TTS_URL): return None
    auth = IAMAuthenticator(IBM_API_KEY)
    tts = TextToSpeechV1(authenticator=auth)
    tts.set_service_url(IBM_TTS_URL)
    return tts

def init_watsonx():
    if not (IBM_API_KEY and IBM_PROJECT_ID): return None
    creds = Credentials(url=IBM_WATSONX_URL, api_key=IBM_API_KEY)
    model = Model(
        model_id="ibm/granite-3-3b-instruct",   # IBM hosted Granite
        credentials=creds,
        project_id=IBM_PROJECT_ID
    )
    return model

watson_stt = init_stt()
watson_tts = init_tts()
watsonx_model = init_watsonx()

# ---------------------------
# Utilities
# ---------------------------
def determine_message_type(msg: str) -> str:
    m = msg.lower()
    if any(k in m for k in ["budget","spending","expense"]): return "budget_summary"
    if any(k in m for k in ["invest","stock","portfolio"]): return "investment_advice"
    if any(k in m for k in ["tax","deduction","irs"]): return "tax_guidance"
    return "text"

def build_context(profile: Profile) -> str:
    if not profile.profile_completed: return ""
    return f"Profile: {profile.user_type}, Age {profile.age}, Income {profile.monthly_income}, Goals {profile.financial_goals}, Risk {profile.risk_tolerance}, Debt {profile.current_debt}"

def call_granite(prompt: str) -> str:
    if not watsonx_model:
        return "IBM Watsonx model not configured."
    try:
        resp = watsonx_model.generate(prompt=prompt, max_new_tokens=512, temperature=0.2)
        return resp['results'][0]['generated_text']
    except Exception as e:
        return f"Granite error: {e}"

def watson_transcribe(file_bytes: bytes, content_type="audio/wav") -> str:
    if not watson_stt: return "STT not configured"
    resp = watson_stt.recognize(audio=io.BytesIO(file_bytes), content_type=content_type).get_result()
    return " ".join([r["alternatives"][0]["transcript"] for r in resp.get("results", [])])

def watson_speak(text: str) -> bytes:
    if not watson_tts: return None
    resp = watson_tts.synthesize(text, voice="en-US_AllisonV3Voice", accept="audio/wav").get_result()
    return resp.content

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("FinanceAI")

# Sidebar Profile
st.sidebar.header("Profile")
profile = load_profile()
with st.sidebar.form("profile_form"):
    profile.user_type = st.selectbox("I am a:", ["","student","professional"], index=1 if profile.user_type=="student" else 2 if profile.user_type=="professional" else 0)
    profile.age = st.number_input("Age", 16, 120, profile.age or 25)
    profile.monthly_income = st.number_input("Monthly Income", 0, 100000, profile.monthly_income or 0)
    profile.current_debt = st.number_input("Current Debt", 0, 1000000, profile.current_debt or 0)
    profile.has_emergency_fund = st.checkbox("Emergency Fund?", profile.has_emergency_fund)
    profile.risk_tolerance = st.selectbox("Risk Tolerance", ["","conservative","moderate","aggressive"], index=(["conservative","moderate","aggressive"].index(profile.risk_tolerance)+1) if profile.risk_tolerance else 0)
    goals = st.text_area("Goals (comma separated)", ",".join(profile.financial_goals or []))
    profile.financial_goals = [g.strip() for g in goals.split(",") if g.strip()]
    profile.profile_completed = st.checkbox("Profile completed", profile.profile_completed)
    if st.form_submit_button("Save"):
        save_profile(profile)
        st.sidebar.success("Saved ✅")

# Chat
st.subheader("Chat")
df = load_messages()
for _, row in df.iterrows():
    who = "You" if row.is_user else "FinanceAI"
    st.markdown(f"**{who}:** {row.message}")

msg = st.text_area("Your message")
if st.button("Send"):
    ts = int(time.time())
    msg_type = determine_message_type(msg)
    save_message_row({"id":ts,"timestamp":ts,"is_user":True,"message":msg,"message_type":msg_type})
    system_prompt = f"You are FinanceAI, a personal financial advisor.\n{build_context(profile)}\nUser: {msg}"
    ai_text = call_granite(system_prompt)
    save_message_row({"id":ts+1,"timestamp":ts+1,"is_user":False,"message":ai_text,"message_type":msg_type})
    st.rerun()
