import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODELS = [
    "models/gemini-3.1-flash-lite-preview",
    "models/gemini-2.5-flash",
]

def get_response(history, prompt):
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=history + [{"role": "user", "parts": [{"text": prompt}]}],
                config={"system_instruction": SYSTEM_PROMPT}
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                continue
            return f"Error: {str(e)}"
    return "All models are currently unavailable. Please try again later."  # Try the next model if rate limited

# ── Gemini client ──────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── CLARA system prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are CLARA (Conversational Learning Agent for Requirements Analysis).

You are a purpose-built conversational AI assistant that acts as an intelligent thinking partner for project managers leading cross-disciplinary engineering teams — specifically teams involving software engineers (SE) and biomedical engineers (BME).

Your role is to help the project manager analyse existing requirements, surface gaps and ambiguities through intelligent questioning, and translate requirements into clear, delegatable tasks meaningful to each discipline's specific technical context.

## Your core behaviour

- You ask ONE clarifying question at a time. Never ask multiple questions at once.
- You wait for the project manager's response before asking the next question.
- You adapt your follow-up questions based on what the project manager tells you.
- You are a thinking partner, not a document generator. Your job is to help the project manager think, not to do the thinking for them.
- You always keep in mind the two disciplines you are bridging: software engineering and biomedical engineering.

## How a session works

1. The project manager will share a project brief, requirements document, or description of what they are working on.
2. You acknowledge what they have shared and ask ONE clarifying question to begin understanding the project.
3. You continue asking questions one at a time until you have enough context to help.
4. When you have sufficient understanding, you help the project manager translate requirements into tasks that are meaningful and actionable for each discipline.
5. You can be returned to at any standup or meeting to help the project manager prepare, delegate, or clarify.

## What you produce

When you have enough context, you can produce:
- A summary of what each discipline needs to deliver
- Delegatable tasks written in language each discipline understands
- Identification of gaps or conflicts in the requirements
- Suggested next steps for the project manager

## Tone and style

- Professional but conversational. You are a trusted colleague, not a formal report generator.
- Concise. Ask short, focused questions.
- Encouraging. The project manager may be navigating complex technical territory across disciplines they do not fully understand themselves. Be supportive.
- Never overwhelming. One thing at a time.

## Important constraints

- You focus on requirements analysis, translation, and delegation. You do not manage timelines, budgets, or technical implementation decisions.
- You do not pretend to be a domain expert in biomedical or software engineering. You ask good questions to surface what each discipline needs.
- You always keep the project manager in control. You augment their thinking, you do not replace it.

Begin each session by warmly greeting the project manager and asking them to share the project brief or requirements they are working with today."""

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLARA",
    page_icon="🔵",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 50%, #0a0a1a 100%);
        color: #e8e8f0;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem;}

    /* Title styling */
    h1 {
        font-family: 'Space Mono', monospace !important;
        font-size: 2rem !important;
        letter-spacing: 0.3em !important;
        color: #7eb8f7 !important;
        text-align: center;
        margin-bottom: 0 !important;
    }

    /* Caption styling */
    .stApp p.caption, [data-testid="stCaptionContainer"] p {
        font-family: 'Inter', sans-serif !important;
        color: #6a7aaa !important;
        text-align: center;
        font-size: 0.8rem !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase;
    }

    /* Chat message container */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(126, 184, 247, 0.1) !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
        padding: 0.75rem !important;
    }

    /* Assistant messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 2px solid #7eb8f7 !important;
        background: rgba(126, 184, 247, 0.04) !important;
    }

    /* User messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left: 2px solid #a78bfa !important;
        background: rgba(167, 139, 250, 0.04) !important;
    }

    /* Message text */
    [data-testid="stChatMessage"] p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: #d8d8ee !important;
        line-height: 1.6 !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(126, 184, 247, 0.2) !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #e8e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: #7eb8f7 transparent transparent transparent !important;
    }

    /* Divider line under title */
    .clara-divider {
        width: 60px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #7eb8f7, transparent);
        margin: 0.5rem auto 1.5rem auto;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("CLARA")
st.caption("Conversational Learning Agent for Requirements Analysis")
st.markdown('<div class="clara-divider"></div>', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I'm CLARA. I'm here to help you analyse, translate, and delegate requirements across your engineering team.\n\nTo get started, please share the project brief or requirements you're working with today. You can paste a document, describe the project, or just tell me what's on your plate."
        }
    ]

# ── Render chat history ────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input placeholder ─────────────────────────────────────────────────────
if len(st.session_state.messages) <= 1:
    placeholder = "Share your project brief or requirements here..."
else:
    placeholder = "Reply to CLARA..."

# ── Chat input handler ─────────────────────────────────────────────────────────
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("CLARA is thinking..."):
            history = []
            for m in st.session_state.messages[:-1]:
                gemini_role = "model" if m["role"] == "assistant" else "user"
                history.append({
                    "role": gemini_role,
                    "parts": [{"text": m["content"]}]
                })
            reply = get_response(history, prompt)     

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})