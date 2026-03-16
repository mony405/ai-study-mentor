"""
AI Study Mentor — main.py
Professional UI: no emojis, editorial design language
"""
from src.config import configure_llm
configure_llm()
import streamlit as st
import asyncio
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Mentor",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — refined, editorial, professional
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 860px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #1f1f1f;
    }
    [data-testid="stSidebar"] * {
        color: #c8c8c8 !important;
    }
    [data-testid="stSidebar"] .stTextInput > div > input,
    [data-testid="stSidebar"] .stFileUploader > div {
        background: #1a1a1a !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #1a1a1a !important;
        border: 1px solid #2e2e2e !important;
        color: #e0e0e0 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
        transition: border-color 0.15s, color 0.15s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #888 !important;
        color: #fff !important;
    }

    /* ── Sidebar headings ── */
    .sidebar-wordmark {
        font-family: 'DM Serif Display', serif;
        font-size: 1.35rem;
        color: #ffffff !important;
        letter-spacing: -0.01em;
        line-height: 1.2;
        margin-bottom: 0.15rem;
    }
    .sidebar-tagline {
        font-size: 0.72rem;
        color: #555 !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #444 !important;
        margin: 1.4rem 0 0.6rem 0;
    }

    /* ── Status badge ── */
    .status-ready {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #22c55e !important;
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 4px;
        padding: 3px 10px;
        margin-top: 0.5rem;
    }
    .status-idle {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #555 !important;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 3px 10px;
        margin-top: 0.5rem;
    }

    /* ── Chat area heading ── */
    .chat-heading {
        font-family: 'DM Serif Display', serif;
        font-size: 1.5rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        color: #f0f0f0;
        margin-bottom: 0.25rem;
    }
    .chat-sub {
        font-size: 0.8rem;
        color: #666;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    /* ── Message bubbles — hide default colored avatars ── */
    [data-testid="stChatMessage"] {
        border-radius: 8px;
        margin-bottom: 0.25rem;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    /* Subtle left border instead of avatar to distinguish speakers */
    [data-testid="stChatMessage"][data-testid*="user"] {
        border-left: 2px solid #444;
        padding-left: 0.75rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 2px solid #6366f1;
        padding-left: 0.75rem;
    }

    /* ── Source / confidence captions ── */
    .source-line {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #999;
        margin-top: 0.4rem;
    }

    /* ── Quiz card ── */
    .quiz-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 0.5rem;
    }
    .quiz-q {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        font-weight: 400;
        line-height: 1.55;
        color: #e8e8e8;
        margin-bottom: 1rem;
    }
    .difficulty-tag {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 3px;
        margin-bottom: 0.75rem;
        background: rgba(99,102,241,0.1);
        color: #6366f1;
    }

    /* ── Summary card ── */
    .summary-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #16a34a;
        margin-bottom: 0.75rem;
    }

    /* ── Shortcut buttons ── */
    .stButton > button {
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-weight: 500 !important;
    }

    /* ── Dividers ── */
    hr {
        border: none;
        border-top: 1px solid #ebebeb;
        margin: 1.5rem 0;
    }

    /* ── Expanders ── */
    details > summary {
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] > button {
        font-size: 0.78rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div {
        background-color: #6366f1 !important;
    }

    /* ── Global accent: replaces Streamlit red on radio + slider + checkbox ── */
    * { accent-color: #6366f1; }

    /* Slider thumb */
    [data-testid="stSlider"] [role="slider"] {
        background-color: #6366f1 !important;
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
    }
    [data-testid="stSlider"] div[data-testid="stThumbValue"] {
        color: #6366f1 !important;
    }

    /* Primary button — replace red with indigo */
    button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background-color: #6366f1 !important;
        border-color: #6366f1 !important;
        color: #fff !important;
    }
    button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: #4f52d4 !important;
        border-color: #4f52d4 !important;
    }

    /* ── Empty state ── */
    .empty-state {
        color: #aaa;
        font-size: 0.9rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "pdf_ready": False,
        "quiz_options_pending": False,
        "quiz_collapsed": False,
        "summary_collapsed": False,
        "subject": "",
        "uploaded_file_name": None,
        "chat_history": [],
        "sidebar_mode": "welcome",
        "summary_text": None,
        "quiz": None,
        "quiz_current_q": 0,
        "quiz_answers": {},
        "quiz_finished": False,
        "workflow": None,
        "index_factory": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ──────────────────────────────────────────────────────────────────────────────
# LLAMAINDEX SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
import os
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

if "llm_ready" not in st.session_state:
    Settings.llm         = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    st.session_state["llm_ready"] = True


# ──────────────────────────────────────────────────────────────────────────────
# DEV MODE
# ──────────────────────────────────────────────────────────────────────────────
DEV_MODE    = True
DEV_SUBJECT = "nlp"
DEV_STORAGE = "./test_storage_chroma"

def _dev_autoload():
    if not st.session_state.pdf_ready:
        try:
            from src.storage.index_manager import IndexFactory
            from src.brain.router import StudyMentorWorkflow
            from src.memory.manager import StudySessionMemory
            import uuid

            factory  = IndexFactory(storage_dir=DEV_STORAGE)
            memory   = StudySessionMemory(session_id=str(uuid.uuid4()))
            workflow = StudyMentorWorkflow(
                index_factory=factory,
                memory=memory,
                tavily_key=os.getenv("TAVILY_API_KEY"),
            )
            st.session_state.index_factory       = factory
            st.session_state.workflow            = workflow
            st.session_state.pdf_ready           = True
            st.session_state.subject             = DEV_SUBJECT
            st.session_state.uploaded_file_name  = f"{DEV_SUBJECT} (dev)"
        except Exception as e:
            st.sidebar.error(f"Dev autoload failed: {e}")

if DEV_MODE:
    _dev_autoload()


# ──────────────────────────────────────────────────────────────────────────────
# ASYNC HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def run_async(coro):
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


async def run_workflow(workflow, **kwargs):
    handler = workflow.run(**kwargs)
    return await handler


async def stream_workflow(workflow, **kwargs):
    from llama_index.core.workflow import StreamEvent
    handler = workflow.run(**kwargs)
    async for event in handler.stream_events():
        if hasattr(event, "delta") and event.delta:
            yield event.delta
    await handler


# ──────────────────────────────────────────────────────────────────────────────
# LATEX NORMALISER
# ──────────────────────────────────────────────────────────────────────────────
def fix_latex(text: str) -> str:
    import re
    text = re.sub(r'\\\[', r'\n$$\n', text)
    text = re.sub(r'\\\]', r'\n$$\n', text)
    text = re.sub(r'\\\(', r'$', text)
    text = re.sub(r'\\\)', r'$', text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-wordmark">Study Mentor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">AI-powered learning assistant</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Material</div>', unsafe_allow_html=True)

        subject_input = st.text_input(
            "Subject",
            placeholder="e.g. Biology, Chapter 3",
            key="subject_input",
            label_visibility="collapsed",
        )
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="pdf_uploader",
            help="Upload your study material as a PDF",
            label_visibility="collapsed",
        )

        upload_btn = st.button(
            "Process Document",
            use_container_width=True,
            disabled=not (uploaded_file and subject_input.strip()),
        )

        if upload_btn and uploaded_file and subject_input.strip():
            handle_pdf_upload(uploaded_file, subject_input.strip())

        if st.session_state.pdf_ready:
            st.markdown(
                f'<div class="status-ready">{st.session_state.subject} — ready</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="status-idle">No document loaded</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-label">Usage</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:0.8rem; line-height:1.8; color:#666;">
Ask any question about your document.<br>
Type <strong style="color:#aaa;">summarize</strong> for a full summary.<br>
Type <strong style="color:#aaa;">quiz me</strong> to test your knowledge.<br>
Out-of-scope questions use web search.
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PDF PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def handle_pdf_upload(uploaded_file, subject: str):
    if uploaded_file.name == st.session_state.uploaded_file_name:
        st.sidebar.warning("This file is already loaded.")
        return

    import tempfile
    from src.ingestion.processor import DocumentProcessor
    from src.storage.index_manager import IndexFactory
    from src.brain.router import StudyMentorWorkflow

    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            factory   = IndexFactory()
            processor = DocumentProcessor()
            nodes     = run_async(processor.process(tmp_path, subject))

            factory.get_or_create_index(nodes, index_type="vector", subject=subject)

            import uuid
            from src.memory.manager import StudySessionMemory
            memory   = StudySessionMemory(session_id=str(uuid.uuid4()))
            workflow = StudyMentorWorkflow(
                index_factory=factory,
                memory=memory,
                tavily_key=os.getenv("TAVILY_API_KEY"),
            )

            st.session_state.index_factory       = factory
            st.session_state.workflow            = workflow
            st.session_state.pdf_ready           = True
            st.session_state.subject             = subject
            st.session_state.uploaded_file_name  = uploaded_file.name
            st.session_state.chat_history        = []
            st.session_state.sidebar_mode        = "welcome"
            st.session_state.summary_text        = None
            st.session_state.quiz                = None
            st.session_state.quiz_current_q      = 0
            st.session_state.quiz_answers        = {}
            st.session_state.quiz_finished       = False

        except Exception as e:
            st.error(f"Failed to process document: {e}")
            return
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    st.success("Document ready.")
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# QUIZ OPTIONS PANEL
# ──────────────────────────────────────────────────────────────────────────────
def render_quiz_options_panel():
    with st.container(border=True):
        st.markdown("**Configure Quiz**")
        num_q = st.select_slider(
            "Number of questions",
            options=[5, 10, 15, 20, 25, 30],
            value=10,
            key="quiz_num_questions",
        )
        difficulty = st.radio(
            "Difficulty",
            options=["Easy", "Medium", "Hard", "Mixed"],
            index=3,
            horizontal=True,
            key="quiz_difficulty",
        )
        col_start, col_cancel = st.columns([1, 1])
        with col_start:
            if st.button("Generate Quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_options_pending = False
                handle_user_input(
                    "quiz me",
                    num_questions=num_q,
                    difficulty=difficulty,
                )
        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state.quiz_options_pending = False
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# INLINE QUIZ RENDERER
# ──────────────────────────────────────────────────────────────────────────────
def render_inline_quiz():
    quiz     = st.session_state.quiz
    q_idx    = st.session_state.quiz_current_q
    total    = len(quiz.questions)
    question = quiz.questions[q_idx]
    answered = q_idx in st.session_state.quiz_answers
    collapsed = st.session_state.get("quiz_collapsed", False)

    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown(
            f'<div class="quiz-label">Quiz — Question {q_idx + 1} of {total}</div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        label = "Hide" if not collapsed else "Show"
        if st.button(label, key="toggle_quiz", use_container_width=True):
            st.session_state.quiz_collapsed = not collapsed
            st.rerun()

    st.progress(q_idx / total)

    if collapsed:
        return

    st.markdown(
        f'<span class="difficulty-tag">{question.difficulty.capitalize()}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="quiz-q">{question.question}</div>', unsafe_allow_html=True)

    options = {opt.label: opt.text for opt in question.options}

    if not answered:
        chosen = st.radio(
            "Select an answer:",
            options=list(options.keys()),
            format_func=lambda k: f"{k}.  {options[k]}",
            key=f"quiz_radio_{q_idx}",
        )
        if st.button("Submit", key=f"submit_{q_idx}", use_container_width=True, type="primary"):
            st.session_state.quiz_answers[q_idx] = chosen
            st.rerun()
    else:
        chosen  = st.session_state.quiz_answers[q_idx]
        correct = question.correct_label
        for opt in question.options:
            if opt.label == correct and opt.label == chosen:
                st.success(f"**{opt.label}.** {opt.text}  — your answer, correct")
            elif opt.label == correct:
                st.success(f"**{opt.label}.** {opt.text}  — correct answer")
            elif opt.label == chosen:
                st.error(f"**{opt.label}.** {opt.text}  — your answer")
            else:
                st.markdown(f"**{opt.label}.** {opt.text}")

        st.info(f"**Explanation:** {question.mentor_explanation}")
        st.divider()

        if q_idx + 1 < total:
            if st.button("Next Question", key=f"next_{q_idx}", use_container_width=True):
                st.session_state.quiz_current_q += 1
                st.rerun()
        else:
            if st.button("View Results", key="see_results", use_container_width=True, type="primary"):
                st.session_state.quiz_finished = True
                _save_quiz_results_to_memory()
                st.rerun()


def _save_quiz_results_to_memory():
    quiz      = st.session_state.quiz
    answers   = st.session_state.quiz_answers
    questions = quiz.questions
    total     = len(questions)
    correct   = sum(1 for i, q in enumerate(questions) if answers.get(i) == q.correct_label)

    lines = [f"## Quiz Results — {correct}/{total} correct\n"]
    for i, q in enumerate(questions):
        chosen     = answers.get(i, "—")
        is_correct = chosen == q.correct_label
        status     = "Correct" if is_correct else f"Incorrect (answered: {chosen}, correct: {q.correct_label})"
        lines.append(f"**Q{i+1}:** {q.question}")
        lines.append(f"→ {status}")
        if not is_correct:
            lines.append(f"Note: {q.mentor_explanation}")
        lines.append("")

    summary_text = "\n".join(lines)

    st.session_state.chat_history.append({
        "role": "mentor",
        "content": summary_text,
        "meta": {"is_quiz_result": True},
    })

    workflow = st.session_state.get("workflow")
    if workflow:
        from llama_index.core.llms import ChatMessage, MessageRole
        memory_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=f"The student just completed a quiz. Results:\n{summary_text}"
        )
        run_async(workflow.memory.memory.aput(memory_msg))


def render_inline_quiz_results():
    quiz          = st.session_state.quiz
    answers       = st.session_state.quiz_answers
    questions     = quiz.questions
    total         = len(questions)
    correct_count = sum(1 for i, q in enumerate(questions) if answers.get(i) == q.correct_label)
    pct           = correct_count / total

    if pct == 1.0:
        st.balloons()
        grade_msg = "Perfect score."
    elif pct >= 0.7:
        grade_msg = "Strong result."
    elif pct >= 0.5:
        grade_msg = "Keep studying."
    else:
        grade_msg = "Review the material and try again."

    collapsed = st.session_state.get("quiz_collapsed", False)
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown('<div class="quiz-label">Quiz Results</div>', unsafe_allow_html=True)
    with col_btn:
        label = "Hide" if not collapsed else "Show"
        if st.button(label, key="toggle_quiz_results", use_container_width=True):
            st.session_state.quiz_collapsed = not collapsed
            st.rerun()

    if collapsed:
        st.markdown(f"**{correct_count}/{total}** — {grade_msg}")
        return

    st.markdown(f"### {correct_count} / {total}  —  {grade_msg}")
    st.progress(pct)
    st.divider()

    for i, q in enumerate(questions):
        chosen     = answers.get(i, "—")
        is_correct = chosen == q.correct_label
        icon       = "+" if is_correct else "−"
        with st.expander(f"[{icon}]  Q{i+1}: {q.question}"):
            for opt in q.options:
                if opt.label == q.correct_label and opt.label == chosen:
                    st.success(f"**{opt.label}.** {opt.text}  — your answer, correct")
                elif opt.label == q.correct_label:
                    st.success(f"**{opt.label}.** {opt.text}  — correct")
                elif opt.label == chosen:
                    st.error(f"**{opt.label}.** {opt.text}  — your answer")
                else:
                    st.markdown(f"**{opt.label}.** {opt.text}")
            st.info(f"**Explanation:** {q.mentor_explanation}")

    if st.button("New Quiz", use_container_width=True):
        st.session_state.quiz          = None
        st.session_state.quiz_current_q = 0
        st.session_state.quiz_answers  = {}
        st.session_state.quiz_finished = False
        st.rerun()


def render_inline_summary():
    summary   = st.session_state.summary_text
    collapsed = st.session_state.get("summary_collapsed", False)

    with st.container(border=True):
        col_title, col_btn = st.columns([5, 1])
        with col_title:
            st.markdown('<div class="summary-label">Document Summary</div>', unsafe_allow_html=True)
        with col_btn:
            label = "Hide" if not collapsed else "Show"
            if st.button(label, key="toggle_summary", use_container_width=True):
                st.session_state.summary_collapsed = not collapsed
                st.rerun()

        if not collapsed:
            st.markdown(fix_latex(summary))
            st.download_button(
                label="Download Summary (.md)",
                data=summary,
                file_name=f"{st.session_state.subject}_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown('<div class="chat-heading">Study Mentor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="chat-sub">'
        f'{"Session active — " + st.session_state.subject if st.session_state.pdf_ready else "Upload a document to begin"}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown('<p class="empty-state">No messages yet.</p>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        render_message(msg)

    if st.session_state.quiz is not None:
        st.divider()
        if st.session_state.quiz_finished:
            render_inline_quiz_results()
        else:
            render_inline_quiz()

    if st.session_state.summary_text is not None:
        st.divider()
        render_inline_summary()

    if st.session_state.get("quiz_options_pending"):
        render_quiz_options_panel()

    if st.session_state.pdf_ready:
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("Summarize", use_container_width=True):
                handle_user_input("summarize")
        with col2:
            if st.button("Quiz Me", use_container_width=True):
                st.session_state.quiz_options_pending = True
                st.rerun()

    user_input = st.chat_input(
        placeholder="Ask a question about your material...",
        disabled=not st.session_state.pdf_ready,
    )
    if user_input:
        handle_user_input(user_input.strip())


def render_message(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    meta    = msg.get("meta", {})

    if role == "student":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(fix_latex(content))

            conf     = meta.get("confidence_score")
            used_web = meta.get("used_web_search", False)
            badges   = []
            if conf is not None and conf > 0:
                if conf >= 0.7:
                    badges.append(f"High confidence ({conf:.0%})")
                elif conf >= 0.4:
                    badges.append(f"Medium confidence ({conf:.0%})")
                else:
                    badges.append(f"Low confidence ({conf:.0%})")
            if used_web:
                badges.append("Web search used")
            if badges:
                st.caption("  ·  ".join(badges))

            sources = meta.get("sources", [])
            if sources:
                lines = []
                for s in sources:
                    m     = s.metadata
                    fname = getattr(m, "file_name", None) or (m.get("file_name", "?") if isinstance(m, dict) else "?")
                    pnum  = getattr(m, "page_number", None) or (m.get("page_number", "?") if isinstance(m, dict) else "?")
                    lines.append(f"{fname} — p. {pnum}")
                st.caption("  ·  ".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# USER INPUT HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def handle_user_input(text: str, num_questions: int = 10, difficulty: str = "Mixed"):
    import queue, concurrent.futures

    st.session_state.chat_history.append({"role": "student", "content": text})

    lower    = text.lower().strip()
    subject  = st.session_state.subject
    workflow = st.session_state.workflow

    if any(kw in lower for kw in ["summarize", "summary", "summarise"]):
        mode = "summary"
        st.session_state.summary_text      = None
        st.session_state.summary_collapsed = False
        text = f"Give me a full summary of the {subject} material."
    elif any(kw in lower for kw in ["quiz", "test me", "quiz me"]):
        mode = "quiz"
        st.session_state.quiz            = None
        st.session_state.quiz_current_q  = 0
        st.session_state.quiz_answers    = {}
        st.session_state.quiz_finished   = False
        st.session_state.quiz_num_q      = num_questions
        st.session_state.quiz_diff       = difficulty
        st.session_state.quiz_collapsed  = False
    else:
        mode = "chat"

    workflow_kwargs = dict(
        query         = text,
        subject       = subject,
        mode          = mode,
        num_questions = st.session_state.get("quiz_num_q", 10),
        difficulty    = st.session_state.get("quiz_diff", "Mixed"),
    )

    if mode in ("quiz", "summary"):
        with st.spinner("Working..."):
            try:
                response = run_async(run_workflow(workflow, **workflow_kwargs))
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "mentor", "content": f"Error: {e}", "meta": {},
                })
                st.rerun()
                return

        if mode == "quiz":
            st.session_state.quiz            = response
            st.session_state.quiz_current_q  = 0
            st.session_state.quiz_answers    = {}
            st.session_state.quiz_finished   = False
            st.session_state.quiz_collapsed  = False
            diff_label = st.session_state.get("quiz_diff", "Mixed")
            st.session_state.chat_history.append({
                "role": "mentor",
                "content": f"Quiz ready — {response.total_questions} questions · {diff_label} difficulty. See below.",
                "meta": {},
            })
        else:
            summary_text = response if isinstance(response, str) else getattr(response, "answer", str(response))
            st.session_state.summary_text      = summary_text
            st.session_state.summary_collapsed = False
            st.session_state.chat_history.append({
                "role": "mentor",
                "content": "Summary generated — see below.",
                "meta": {},
            })
            from llama_index.core.llms import ChatMessage, MessageRole
            run_async(workflow.memory.memory.aput(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"I just generated a full summary of the {subject} material:\n\n{summary_text}"
                )
            ))

        st.rerun()
        return

    try:
        response      = run_async(run_workflow(workflow, **workflow_kwargs))
        answer        = response if isinstance(response, str) else getattr(response, "answer", str(response))
        answer        = fix_latex(answer)
        words         = answer.split(" ")

        def _word_gen():
            for word in words:
                yield word + " "

        with st.chat_message("assistant"):
            streamed_text = st.write_stream(_word_gen())

    except Exception as e:
        st.session_state.chat_history.append({
            "role": "mentor", "content": f"Error: {e}", "meta": {},
        })
        st.rerun()
        return

    if isinstance(response, str):
        meta = {}
    else:
        meta = {
            "confidence_score": getattr(response, "confidence_score", None),
            "used_web_search":  getattr(response, "used_web_search", False),
            "sources":          getattr(response, "sources", []),
        }
    st.session_state.chat_history.append({
        "role": "mentor", "content": streamed_text, "meta": meta,
    })

    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()