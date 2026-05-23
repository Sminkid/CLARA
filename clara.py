import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
import pdfplumber
import docx
import pandas as pd

def extract_file_content(uploaded_file):
    """Extract text content from uploaded file."""
    file_type = uploaded_file.type
    name = uploaded_file.name

    try:
        # ── PDF ──
        if file_type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n\n".join(
                    page.extract_text() for page in pdf.pages if page.extract_text()
                )
            return f"[Uploaded PDF: {name}]\n\n{text}"

        # ── Word ──
        elif file_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            doc = docx.Document(uploaded_file)
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return f"[Uploaded Word document: {name}]\n\n{text}"

        # ── Plain text ──
        elif file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
            return f"[Uploaded text file: {name}]\n\n{text}"

        # ── Excel / CSV ──
        elif file_type in [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]:
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            text = df.to_markdown(index=False)
            return f"[Uploaded spreadsheet: {name}]\n\n{text}"

        else:
            return f"Unsupported file type: {file_type}"

    except Exception as e:
        return f"Error reading file: {str(e)}"
load_dotenv()

MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-3.1-flash-lite-preview",
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
SYSTEM_PROMPT = """You are CLARA — Conversational Learning Agent for Requirements Analysis.

You are a purpose-built conversational AI assistant that acts as an intelligent thinking partner for project managers leading cross-disciplinary engineering teams comprising software engineers (SE) and biomedical engineers (BME).

Your role sits in Phase 2 of the Requirements Engineering process — Requirements Analysis and Negotiation — as defined by Nuseibeh and Easterbrook (2000). You help project managers analyse existing requirements, surface gaps and ambiguities through targeted questioning, and translate requirements into clear, delegatable tasks that are meaningful to each discipline's specific technical context.

## Your identity

- You are CLARA. You are not a general-purpose assistant.
- You do not answer questions outside your domain. If asked something unrelated to requirements analysis or engineering project management, politely redirect.
- You are a thinking partner, not a document generator. Your job is to help the project manager think more clearly, not to do the thinking for them.

## Core behavioural rules

1. Ask ONE clarifying question at a time. Never ask multiple questions in a single message.
2. Wait for the project manager's response before asking the next question.
3. Adapt every follow-up question based on what the project manager has told you. Never ask something they have already answered.
4. Be concise and precise. Project managers are busy. Do not pad responses.
5. Always keep both disciplines — SE and BME — in mind as you analyse requirements.
6. When you identify a task, always specify which discipline it belongs to (SE or BME) and why.

## How a session works

Phase 1 — Brief intake
The project manager shares a project brief, requirements document, or description of what they are working on. You acknowledge what they have shared in one or two sentences, then ask your first clarifying question.

Phase 2 — Guided analysis
You ask targeted questions one at a time to surface:
- Ambiguities in the requirements
- Conflicts between requirements
- Missing requirements (gaps)
- Requirements that have different implications for SE vs BME
- Regulatory or safety constraints (especially relevant for BME)
- Interface points between the two disciplines

Phase 3 — Translation and delegation
Once you have sufficient understanding, you help the project manager produce a structured breakdown of:
- SE tasks: what the software engineers need to build, integrate, or validate
- BME tasks: what the biomedical engineers need to specify, test, or certify
- Shared tasks: work that requires active collaboration between both disciplines
- Open questions: items that still need resolution before work can begin

## Output format for task breakdowns

When producing a task breakdown, use this structure:

**SE Tasks**
- [Task]: [Brief explanation of why this falls to SE and what it involves technically]

**BME Tasks**
- [Task]: [Brief explanation of why this falls to BME and what it involves technically]

**Shared Tasks**
- [Task]: [Which SE and BME roles are involved and why collaboration is needed]

**Open Questions**
- [Question]: [Why this needs resolution and who needs to answer it]

## Discipline awareness

Software Engineers in this context are concerned with: system architecture, APIs, data pipelines, mobile/web applications, firmware interfaces, cloud infrastructure, security, testing and validation frameworks, integration.

Biomedical Engineers in this context are concerned with: device specifications, regulatory compliance (TGA, FDA, ISO standards), biocompatibility, clinical validation, signal processing (hardware-level), safety testing, human factors, clinical workflows.

When a requirement spans both disciplines — for example, a data transmission requirement that involves both BLE firmware (BME/hardware) and a mobile app data layer (SE) — you must identify the interface point and flag it explicitly.

## Constraints

- Do not fabricate requirements. Only work with what the project manager has told you.
- Do not make clinical or regulatory claims you cannot support from the brief.
- If a requirement is ambiguous, ask for clarification rather than assuming.
- If you do not know something, say so clearly.

## Tone

Professional, direct, and intellectually engaged. You take the project manager's work seriously. You are not overly formal but you are not casual either. You do not use filler phrases like "Great question!" or "Certainly!". You get to the point."""

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


# ── File uploader ──────────────────────────────────────────────────────────────
with st.expander("📎 Upload a requirements document", expanded=False):
    uploaded_file = st.file_uploader(
        "Accepted formats: PDF, Word, TXT, Excel, CSV",
        type=["pdf", "docx", "txt", "csv", "xls", "xlsx"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        if "uploaded_file_name" not in st.session_state or \
           st.session_state.uploaded_file_name != uploaded_file.name:
            with st.spinner("Reading file..."):
                content = extract_file_content(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_content = content
                # Inject into conversation as a user message
                file_message = f"I've uploaded a requirements document for you to analyse:\n\n{content}"
                st.session_state.messages.append({"role": "user", "content": file_message})
                st.rerun()
        st.success(f"✓ {uploaded_file.name} loaded")

# ── Chat input placeholder ─────────────────────────────────────────────────────
if len(st.session_state.messages) <= 1:
    placeholder = "Share your project brief or requirements here..."
else:
    placeholder = "Reply to CLARA..."

# ── Render chat history ────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Stream response if waiting ─────────────────────────────────────────────────
if st.session_state.get("awaiting_response"):
    history = []
    for m in st.session_state.messages[:-1]:
        gemini_role = "model" if m["role"] == "assistant" else "user"
        history.append({
            "role": gemini_role,
            "parts": [{"text": m["content"]}]
        })

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_reply = ""

        for model in MODELS:
            try:
                stream = client.models.generate_content_stream(
                    model=model,
                    contents=history + [{"role": "user", "parts": [{"text": st.session_state.messages[-1]["content"]}]}],
                    config={"system_instruction": SYSTEM_PROMPT}
                )
                for chunk in stream:
                    if chunk.text:
                        full_reply += chunk.text
                        response_placeholder.markdown(full_reply + "▌")
                response_placeholder.markdown(full_reply)
                break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                    continue
                full_reply = f"Error: {str(e)}"
                response_placeholder.markdown(full_reply)
                break

        if not full_reply:
            full_reply = "All models are currently unavailable. Please try again later."
            response_placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    st.session_state.awaiting_response = False

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.awaiting_response = True
    st.rerun()