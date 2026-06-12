CLARA — Conversational Learning Agent for Requirements Analysis
CLARA is a purpose-built conversational AI agent designed to act as a thinking partner for project managers leading cross-disciplinary engineering teams. It sits in Phase 2 of the Requirements Engineering process — Requirements Analysis and Negotiation — and helps project managers analyse, translate, and delegate requirements into discipline-specific delegatable tasks.
Research Context
CLARA was developed as part of an honours research project at the University of Technology Sydney, Faculty of Engineering and Information Technology.
Research question: Can a purpose-built conversational AI agent effectively support project managers in analysing, translating, and delegating requirements across cross-disciplinary engineering teams?
Supervisor: Dr Madhushi Bandara, University of Technology Sydney
Methodology: Design Science Research Methodology (Peffers et al. 2007)
This repository accompanies the position paper:

Chang, J. and Bandara, M. (2026). Conversational AI as a Cross-Disciplinary Thinking Partner: Addressing the Technical-Operational Gap in AI Delivery. In: CBI & EDOC 2026 Industry & Impact Track, University of Twente, Enschede, Netherlands.

How It Works
CLARA operates through a three-phase session structure:

Phase 1 — Brief intake: The project manager shares a project brief or requirements document
Phase 2 — Guided analysis: CLARA asks one targeted clarifying question at a time to surface ambiguities, conflicts, gaps, and interface points between disciplines
Phase 3 — Translation and delegation: CLARA produces a structured task breakdown with discipline-specific tasks, shared tasks, and open questions

Technical Stack

Python
Streamlit
Google Gemini API (gemini-2.5-flash primary, gemini-3.1-flash-lite-preview fallback)
google-genai
python-dotenv

Setup

Clone the repository
Install dependencies:

pip install -r requirements.txt

Create a .env file in the root directory with your Gemini API key:

GEMINI_API_KEY=your_api_key_here

Run the app:

streamlit run clara.py
Research Artefacts
The following files are included to support reproducibility of the research reported in the paper:

system_prompt.txt — the full engineered system prompt used in CLARA v1, encoding the three-phase session structure, behavioural constraints, and discipline-specific translation rules
smartpatch_brief.txt — the fabricated cross-disciplinary medical device brief used as the primary test input across iterative test sessions
The NASA/US Army Tactical Control System specification used in additional testing is drawn from the publicly available PURE dataset (Ferrari et al. 2017) and is not redistributed here

Status
CLARA is currently under active development as part of an ongoing honours research project. The prototype is at v1 and a formal evaluation with industry project managers is planned for August 2026. Findings are preliminary and represent feasibility evidence rather than validated results.
Citation
If you use or build on this work please cite:

Chang, J. and Bandara, M. (2026). Conversational AI as a Cross-Disciplinary Thinking Partner: Addressing the Technical-Operational Gap in AI Delivery. CBI & EDOC 2026 Industry & Impact Track.

Acknowledgements
This research is supervised by Dr Madhushi Bandara at the University of Technology Sydney. Development was supported by the Google Gemini API.
