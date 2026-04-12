import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def get_client():
    api_key = get_api_key()
    return Groq(api_key=api_key) if api_key else None


def safe_get(d, *keys, default="N/A"):
    try:
        for k in keys:
            d = d[k]
        return d
    except:
        return default


def fallback_summary(a):
    return f"""
Player: {safe_get(a,'prediction','label')}
Confidence: {safe_get(a,'confidence')}%

Risk: {safe_get(a,'risk','level')}
Loss Risk: {safe_get(a,'loss_prediction','level')}

Worst Loss: ₹{safe_get(a,'loss_prediction','projected_worst_case')}

Strategy: {safe_get(a,'strategy','recommendation')}
Future: {safe_get(a,'simulation','trend')}

Score: {safe_get(a,'score')}

Final Action: {safe_get(a,'final_action')}
"""


def generate_agent_summary(analysis):
    client = get_client()

    if client is None:
        return fallback_summary(analysis)

    prompt = f"""
Analyze casino player:

Player Type: {safe_get(analysis,'prediction','label')}
Confidence: {safe_get(analysis,'confidence')}

Risk Level: {safe_get(analysis,'risk','level')}
Loss Risk Level: {safe_get(analysis,'loss_prediction','level')}

Worst Case Loss: {safe_get(analysis,'loss_prediction','projected_worst_case')}

Strategy: {safe_get(analysis,'strategy','recommendation')}
Future Trend: {safe_get(analysis,'simulation','trend')}

Score: {safe_get(analysis,'score')}

Final Action: {safe_get(analysis,'final_action')}

Explain clearly and professionally.
"""

    try:
        res = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content

    except Exception:
        return fallback_summary(analysis)
