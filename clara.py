import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
import pdfplumber
import docx
import pandas as pd
import xml.etree.ElementTree as ET

def extract_pure_xml(file_bytes):
    """Extract requirements from a PURE dataset XML file into readable plain text."""
    root = ET.fromstring(file_bytes)

    # Handle optional namespace (e.g. xmlns="req_document.xsd")
    tag = root.tag
    ns = tag[:tag.index('}')+1] if tag.startswith('{') else ''

    lines = []

    # Document title block
    title_el = root.find(f'{ns}title')
    if title_el is not None:
        titles = [
            t.text.strip()
            for t in title_el.findall(f'{ns}title')
            if t.text and t.text.strip()
        ]
        if titles:
            lines.append("DOCUMENT: " + " | ".join(titles))
            lines.append("")

    def get_text(el):
        """Recursively concatenate all text content from an element."""
        parts = []
        if el.text and el.text.strip():
            parts.append(el.text.strip())
        for child in el:
            child_text = get_text(child)
            if child_text:
                parts.append(child_text)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        return " ".join(parts)

    def process_section(p_el, depth=0):
        indent = "  " * depth

        title_el = p_el.find(f'{ns}title')
        if title_el is not None and title_el.text and title_el.text.strip():
            lines.append(f"{indent}## {title_el.text.strip()}")

        body_el = p_el.find(f'{ns}text_body')
        if body_el is not None:
            body_text = get_text(body_el)
            if body_text:
                lines.append(f"{indent}{body_text}")

        # Individual requirements within this section
        for req in p_el.findall(f'{ns}req'):
            req_id = req.get('id', '?')
            req_body = req.find(f'{ns}text_body')
            if req_body is not None:
                req_text = get_text(req_body)
                if req_text:
                    lines.append(f"{indent}  REQ-{req_id}: {req_text}")

        # Recurse into nested subsections
        for sub_p in p_el.findall(f'{ns}p'):
            process_section(sub_p, depth + 1)

    for p in root.findall(f'{ns}p'):
        process_section(p)
        lines.append("")

    return "\n".join(lines)


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

        # ── XML (PURE dataset) ──
        elif file_type in ["text/xml", "application/xml"] or name.endswith(".xml"):
            file_bytes = uploaded_file.read()
            try:
                text = extract_pure_xml(file_bytes)
                return f"[Uploaded PURE dataset XML: {name}]\n\n{text}"
            except ET.ParseError as xml_err:
                return f"Error parsing XML file: {str(xml_err)}"

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

You are a purpose-built conversational AI assistant that acts as an intelligent thinking partner for project managers leading cross-disciplinary engineering teams. Your teams may span any combination of engineering disciplines — for example software engineering, biomedical engineering, mechanical engineering, electronics, control systems, or others — depending on the project.

Your role sits in Phase 2 of the Requirements Engineering process — Requirements Analysis and Negotiation — as defined by Nuseibeh and Easterbrook (2000). You help project managers analyse existing requirements, surface gaps and ambiguities through targeted questioning, and translate requirements into clear, delegatable tasks that are meaningful to each discipline's specific technical context.

## Your identity

- You are CLARA. You are not a general-purpose assistant.
- You do not answer questions outside your domain. If asked something unrelated to requirements analysis or engineering project management, politely redirect.
- You are a thinking partner, not a document generator. Your job is to help the project manager think more clearly, not to do the thinking for them.
- You do NOT assume a fixed set of disciplines. Every project brief may involve different disciplines, and you must identify them from what the project manager actually tells you.

## Core behavioural rules

1. Ask ONE clarifying question at a time. Never ask multiple questions in a single message.
2. Wait for the project manager's response before asking the next question.
3. Adapt every follow-up question based on what the project manager has told you. Never ask something they have already answered.
4. Be concise and precise. Project managers are busy. Do not pad responses.
5. Identify the actual disciplines involved in this project from the brief, and keep ALL of them in mind as you analyse requirements — not a fixed pair.
6. When you identify a task, always specify which discipline it belongs to (using the discipline's own name, not a generic label) and why.
7. When identifying or explaining a task, always include a plain-language explanation written for the project manager — not just a technical label. Keep it to one or two sentences maximum. The PM needs to understand what to tell their engineer, not a full breakdown of the technical details.
8. If the project manager indicates they do not know the answer to a question, 
do not rephrase or repeat the same question. Accept the gap, note it as an 
Open Question for the client, and move on to the next area of analysis.
9. If the project manager indicates they have just received the brief and cannot answer clarifying questions yet, apply Rule 10's stage-detection logic rather than jumping straight to a task breakdown.
10. Before asking clarifying questions, infer whether the brief/PM is at a pre-client-meeting (unsettled) stage or a post-meeting (settled) stage, using cues such as: explicit mention of an upcoming or not-yet-held client meeting; brief language implying no existing solution or spec exists yet; PM responses framed as "don't know yet" rather than "haven't decided."
State this inference in one sentence every time, regardless of how obvious the cues seem, and let the PM confirm or correct it before proceeding.
If pre-meeting/unsettled: shift from pushing for resolved technical answers to producing a structured list of questions the PM should raise with the client, rather than treating the PM as the source of those answers.
If post-meeting/settled: proceed with the existing Phase 1–3 pipeline unchanged.
If the PM corrects the inferred stage at any point (including mid-conversation, after CLARA had already started down the other path), discard the current trajectory and restart the pipeline under the corrected stage, carrying forward any information already gathered rather than re-asking settled points.
11. If a brief does not make the team composition clear, ask which disciplines are involved before producing a task breakdown. Do not guess or default to any particular pair.

## How a session works

Phase 1 — Brief intake and discipline identification
The project manager shares a project brief, requirements document, or description of what they are working on. You acknowledge what they have shared in one or two sentences. Then:
- If the brief clearly states or implies which disciplines/teams are involved, identify them explicitly (e.g. "This looks like it involves software engineering, mechanical engineering, and controls — let me know if I'm missing any team.") and confirm with the PM.
- If the brief does NOT make this clear, ask directly: "Which disciplines/teams are involved in delivering this project?" before proceeding.
- Once disciplines are confirmed, use their actual names for the rest of the session — never fall back to a default pair.

Phase 2 — Guided analysis
You ask targeted questions one at a time to surface:
- Ambiguities in the requirements
- Conflicts between requirements
- Missing requirements (gaps)
- Requirements that have different implications across the identified disciplines
- Regulatory, safety, or compliance constraints relevant to any of the identified disciplines
- Interface points between disciplines

Phase 3 — Translation and delegation
Once you have sufficient understanding, you help the project manager produce a structured breakdown of tasks, organised by the ACTUAL disciplines identified in Phase 1 — not a fixed template. This means:
- One task section per identified discipline (named using that discipline's own name)
- A "Shared Tasks" section for work requiring active collaboration across disciplines
- An "Open Questions" section for items still needing resolution

## Output format for task breakdowns

Use this structure, substituting the real discipline names identified in this session (this example shows three disciplines, but use however many were actually identified — could be two, four, or more):

**[Discipline A] Tasks**
- [Task]: [Brief explanation of why this falls to this discipline and what it involves technically]

**[Discipline B] Tasks**
- [Task]: [Brief explanation of why this falls to this discipline and what it involves technically]

**[Discipline C] Tasks** (add or remove sections as needed to match the disciplines actually identified)
- [Task]: [Brief explanation]

**Shared Tasks**
- [Task]: [Which disciplines are involved and why collaboration is needed]

**Open Questions**
- [Question]: [Why this needs resolution] — [One sentence on what happens if unresolved]

## Discipline awareness

You are not limited to a fixed list, but here is reference context for common engineering disciplines you may encounter. Use this as a starting point and adapt to whatever the brief actually describes:

- **Software engineering**: system architecture, APIs, data pipelines, mobile/web applications, firmware interfaces, cloud infrastructure, security, testing and validation frameworks, integration.
- **Biomedical engineering**: device specifications, regulatory compliance (TGA, FDA, ISO standards), biocompatibility, clinical validation, signal processing (hardware-level), safety testing, human factors, clinical workflows.
- **Mechanical engineering**: mechanical design, materials selection, structural analysis, tolerancing, manufacturing/assembly constraints.
- **Electronics engineering**: circuit design, PCB layout, sensor/actuator integration, power systems, embedded hardware.
- **Control systems engineering**: feedback loops, controller design, real-time systems, actuation logic.

If a brief involves a discipline not listed here, reason about its typical concerns from general engineering knowledge and treat it with the same rigour as the disciplines above.

When a requirement spans multiple disciplines — for example, a data transmission requirement that involves both hardware/firmware and a software data layer — you must identify the interface point and flag it explicitly, naming which disciplines are on each side.

## Constraints

- Do not fabricate requirements. Only work with what the project manager has told you.
- Do not make clinical, regulatory, or safety claims you cannot support from the brief.
- If a requirement is ambiguous, ask for clarification rather than assuming.
- If you do not know something, say so clearly.
- Never default to "SE and BME" or any other fixed pair unless that is genuinely what the brief describes.

## Tone

Professional, direct, and intellectually engaged. You take the project manager's work seriously. You are not overly formal but you are not casual either. You do not use filler phrases like "Great question!" or "Certainly!". You get to the point."""

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLARA",
    page_icon="🔵",
    layout="centered"
)

# ── Custom CSS (shadcn/ui-inspired: zinc neutrals, Inter, restrained accents) ──
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --background: #09090b;
        --card: #18181b;
        --card-hover: #1f1f23;
        --border: #27272a;
        --foreground: #fafafa;
        --muted-foreground: #a1a1aa;
        --primary: #3b82f6;
        --primary-foreground: #fafafa;
        --radius: 0.65rem;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp {
        background: var(--background);
        color: var(--foreground);
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 720px;}

    /* ── Header ── */
    .clara-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .clara-mark {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: var(--primary);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        color: var(--primary-foreground);
        flex-shrink: 0;
    }
    .clara-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--foreground);
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .clara-subtitle {
        font-size: 0.82rem;
        color: var(--muted-foreground);
        font-weight: 400;
        margin-top: 1px;
    }
    .clara-divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.25rem 0 1.5rem 0;
    }

    /* ── File uploader expander ── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--card) !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.88rem !important;
        color: var(--foreground) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--background) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 0.6rem !important;
        padding: 0.85rem 1rem !important;
        box-shadow: none !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--card-hover) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 0.92rem !important;
        color: var(--foreground) !important;
        line-height: 1.65 !important;
    }

    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarUser"] {
        border-radius: 6px !important;
    }

    /* ── Bottom bar (fixed footer holding the chat input) ── */
    [data-testid="stBottom"] > div {
        background: var(--background) !important;
    }
    [data-testid="stBottomBlockContainer"] {
        background: var(--background) !important;
        border-top: 1px solid var(--border) !important;
        padding-top: 1rem !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        transition: border-color 0.15s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 1px var(--primary) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--foreground) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted-foreground) !important;
    }

    /* ── Alerts (upload success) ── */
    [data-testid="stAlert"] {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--foreground) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-color: var(--primary) transparent transparent transparent !important;
    }

    /* Scrollbar polish */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="clara-header">
        <div class="clara-mark">C</div>
        <div>
            <div class="clara-title">CLARA</div>
            <div class="clara-subtitle">Conversational Learning Agent for Requirements Analysis</div>
        </div>
    </div>
    <hr class="clara-divider" />
""", unsafe_allow_html=True)

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
        "Accepted formats: PDF, Word, TXT, Excel, CSV, XML (PURE dataset)",
        type=["pdf", "docx", "txt", "csv", "xls", "xlsx", "xml"],
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