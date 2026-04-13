import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
import tempfile

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "vector_store": None,
    "vector_store_loaded": False,
    "chat_history": [],
    "plan": [],
    "current_step": 0,
    # phase values:
    #   idle          → user types a topic to study
    #   await_explain → plan shown, asking "shall I explain?"
    #   await_quiz    → explanation done, asking "shall I quiz you?"
    #   await_answer  → quiz question shown, waiting for A/B/C/D
    #   await_next    → graded, asking "ready for next topic?"
    #   done          → all topics complete
    "phase": "idle",
    "last_question": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LLM & Wikipedia ───────────────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

@st.cache_resource
def get_wikipedia():
    # top_k_results=2 keeps it fast; doc_content_chars_max limits token usage
    return WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)

# ── PDF helpers ───────────────────────────────────────────────────────────────
def load_pdfs(uploaded_files):
    all_docs = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in uploaded_files:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as fh:
                fh.write(f.getbuffer())
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())

    if not all_docs:
        return None, "No content found in PDFs"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
    vs = FAISS.from_documents(texts, embeddings)
    return vs, f"Processed {len(uploaded_files)} PDF(s) — {len(all_docs)} pages indexed"

def retrieve_pdf(query: str, k: int = 4) -> str:
    """Pull top-k chunks from the vector store."""
    if not st.session_state.vector_store:
        return ""
    docs = st.session_state.vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)

def retrieve_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary."""
    try:
        return get_wikipedia().run(query)
    except Exception:
        return ""

# ── Is this a free-form question? ─────────────────────────────────────────────
# Simple heuristic: if the message contains a question mark OR is longer than
# 6 words and doesn't match known short commands, treat it as a chatbot question.
YES_WORDS  = {"yes", "y", "sure", "ok", "okay", "go ahead", "yep", "yeah", "please"}
SKIP_WORDS = {"no", "n", "skip", "next"}
CMD_WORDS  = {"retry", "retry quiz", "try again", "re-explain", "explain",
              "re explain", "explain again", "a", "b", "c", "d"}

def is_free_question(text: str, phase: str) -> bool:
    """Return True if we should treat this as a chatbot question rather than a phase command."""
    lowered = text.strip().lower()
    words = lowered.split()

    # Single-letter answers in quiz phase — definitely not a question
    if phase == "await_answer" and lowered in {"a", "b", "c", "d"}:
        return False

    # Known command words → not a question
    if lowered in YES_WORDS | SKIP_WORDS | CMD_WORDS:
        return False

    # Has a question mark → definitely a question
    if "?" in text:
        return True

    # Starts with an interrogative word
    interrogatives = ("what", "who", "when", "where", "why", "how", "which",
                      "explain", "define", "tell me", "describe", "can you")
    if any(lowered.startswith(w) for w in interrogatives):
        return True

    # Long message (>6 words) that isn't a known command → probably a question
    if len(words) > 6:
        return True

    return False

# ── Chatbot answer (PDF + Wikipedia) ─────────────────────────────────────────
def run_chatbot(question: str) -> str:
    pdf_context  = retrieve_pdf(question, k=4)
    wiki_context = retrieve_wikipedia(question)

    sources_used = []
    source_block = ""
    if pdf_context:
        sources_used.append("the uploaded PDF")
        source_block += f"\n\n📄 **From your PDF:**\n{pdf_context}"
    if wiki_context:
        sources_used.append("Wikipedia")
        source_block += f"\n\n🌐 **From Wikipedia:**\n{wiki_context}"

    source_label = " and ".join(sources_used) if sources_used else "general knowledge"

    prompt = f"""You are a helpful study assistant. A student asked: "{question}"

Answer using the following sources ({source_label}):
{source_block if source_block else "No external sources found — use your general knowledge."}

Give a clear, accurate, well-structured answer.
If the PDF has relevant info, prefer it. Use Wikipedia to fill gaps or give broader context.
At the end, add a one-line note like: "📚 Source: PDF" / "🌐 Source: Wikipedia" / "📚🌐 Sources: PDF + Wikipedia" depending on what you used."""

    return get_llm().invoke(prompt).content.strip()

# ── Study node functions ──────────────────────────────────────────────────────

def run_planner(user_topic: str) -> list:
    pdf_context  = retrieve_pdf(user_topic, k=5)
    wiki_context = retrieve_wikipedia(user_topic)
    combined = ""
    if pdf_context:
        combined += f"PDF CONTENT:\n{pdf_context}\n\n"
    if wiki_context:
        combined += f"WIKIPEDIA SUMMARY:\n{wiki_context}"

    prompt = f"""You are a study planner. A student wants to study: "{user_topic}"

Here is reference material:
{combined}

Create exactly 3 concise sub-topic names for a step-by-step study session.
Return ONLY the 3 names separated by commas, nothing else.
Example format: Topic One, Topic Two, Topic Three"""

    response = get_llm().invoke(prompt)
    raw = response.content.strip()
    raw = re.sub(r"^\d+[\.\)]\s*", "", raw, flags=re.MULTILINE)
    topics = [t.strip().rstrip(".") for t in raw.split(",") if t.strip()][:3]
    if len(topics) < 2:
        topics = [t.strip().lstrip("0123456789.) ").rstrip(".") for t in raw.split("\n") if t.strip()][:3]
    return topics

def run_explainer(topic: str) -> str:
    pdf_context  = retrieve_pdf(topic, k=4)
    wiki_context = retrieve_wikipedia(topic)

    prompt = f"""You are a helpful study tutor. Explain "{topic}" to a student.

Use the following sources:

📄 PDF CONTENT:
{pdf_context or "Not available."}

🌐 WIKIPEDIA:
{wiki_context or "Not available."}

Prefer the PDF content where available. Use Wikipedia to add broader context or fill gaps.
Write a clear, structured explanation with key points and examples.
End your response with exactly this line:
✅ Done! Shall I quiz you on this topic? (yes / no)"""

    return get_llm().invoke(prompt).content.strip()

def run_quizzer(topic: str) -> str:
    pdf_context  = retrieve_pdf(topic, k=3)
    wiki_context = retrieve_wikipedia(topic)

    prompt = f"""You are a quiz generator. Create ONE multiple-choice question (A/B/C/D)
to test a student on the topic "{topic}".

Base the question on:

📄 PDF:
{pdf_context or "Not available."}

🌐 Wikipedia:
{wiki_context or "Not available."}

Format:
Question: [your question here]
A) [option]
B) [option]
C) [option]
D) [option]

Show only the question and options. Do NOT reveal the correct answer."""

    return get_llm().invoke(prompt).content.strip()

def run_grader(topic: str, question: str, user_answer: str) -> tuple:
    pdf_context  = retrieve_pdf(topic, k=3)
    wiki_context = retrieve_wikipedia(topic)

    prompt = f"""You are a quiz grader.

Topic: {topic}

Reference material:
📄 PDF: {pdf_context or "Not available."}
🌐 Wikipedia: {wiki_context or "Not available."}

Quiz question: {question}
Student's answer: {user_answer}

Grade the answer. Start with exactly "CORRECT ✅" or "INCORRECT ❌", then explain
why and give the correct answer with a brief explanation."""

    text = get_llm().invoke(prompt).content.strip()
    is_correct = text.upper().startswith("CORRECT")
    return is_correct, text

# ── Chat helpers ──────────────────────────────────────────────────────────────

def add_msg(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

def show_msg(role, content):
    with st.chat_message(role):
        st.markdown(content)

def bot(content):
    add_msg("assistant", content)
    show_msg("assistant", content)

def advance_to_next_topic():
    step = st.session_state.current_step
    if step >= len(st.session_state.plan):
        bot("🎉 **You've completed all topics! Excellent work!**\n\nType a new topic to study, or reset from the sidebar.")
        st.session_state.phase = "done"
    else:
        topic = st.session_state.plan[step]
        bot(f"Moving on to **{topic}**. Shall I explain it? (yes / no)")
        st.session_state.phase = "await_explain"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")
        with st.expander("View files"):
            for f in uploaded_files:
                st.text(f"• {f.name}")

    if st.button("🔄 Process PDFs", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Indexing PDFs…"):
            vs, msg = load_pdfs(uploaded_files)
            if vs:
                st.session_state.vector_store = vs
                st.session_state.vector_store_loaded = True
                st.session_state.chat_history = []
                st.session_state.plan = []
                st.session_state.current_step = 0
                st.session_state.phase = "idle"
                st.success(msg)
            else:
                st.error(msg)

    st.markdown("---")
    if st.session_state.vector_store_loaded:
        st.success("✅ System Ready")
        if st.session_state.plan:
            st.markdown("**Study Progress:**")
            for i, topic in enumerate(st.session_state.plan):
                icon = "✅" if i < st.session_state.current_step else ("▶️" if i == st.session_state.current_step else "⏳")
                st.markdown(f"{icon} {topic}")
    else:
        st.warning("⚠️ Upload PDFs to start")

    if st.button("🔄 Reset Session", use_container_width=True):
        for k in ["chat_history", "plan", "last_question"]:
            st.session_state[k] = DEFAULTS[k]
        st.session_state.current_step = 0
        st.session_state.phase = "idle"
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📚 Lyra : Learn Smarter with AI")
st.markdown("---")

if not st.session_state.vector_store_loaded:
    st.info("👈 Upload your PDF study materials in the sidebar to begin.")
    st.markdown("""
**How it works:**
1. Upload your PDF notes or textbook
2. Click **Process PDFs**
3. Tell the assistant what you want to study
4. It explains each sub-topic step by step and quizzes you before moving on
5. Ask any question at any time — it will search your PDF + Wikipedia to answer
""")
else:
    st.subheader("💬 Study Session")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    phase = st.session_state.phase

    PLACEHOLDERS = {
        "idle":         "What topic do you want to study from your PDF?",
        "await_explain":"'yes' to explain, 'no' to skip — or ask any question!",
        "await_quiz":   "'yes' for a quiz, 'no' to move on — or ask any question!",
        "await_answer": "Type your answer: A, B, C, or D — or ask a question!",
        "await_next":   "'yes' to continue, 'retry', 're-explain' — or ask anything!",
        "done":         "Type a new topic to study, or ask any question!",
    }
    user_input = st.chat_input(PLACEHOLDERS.get(phase, "Ask anything…"))

    if user_input:
        show_msg("user", user_input)
        add_msg("user", user_input)
        lowered = user_input.strip().lower()
        YES = {"yes", "y", "sure", "ok", "okay", "go ahead", "yep", "yeah", "please"}

        with st.spinner("Thinking…"):

            # ── FREE QUESTION: answer using PDF + Wikipedia, don't change phase ──
            if is_free_question(user_input, phase) and phase not in ("idle", "done"):
                answer = run_chatbot(user_input)
                # After answering, remind user where they are
                current_topic = (
                    st.session_state.plan[st.session_state.current_step]
                    if st.session_state.plan and st.session_state.current_step < len(st.session_state.plan)
                    else None
                )
                reminder = f"\n\n---\n_Continuing your study session on **{current_topic}**…_" if current_topic else ""
                bot(answer + reminder)
                # phase stays the same — user can still continue the study flow

            # ── IDLE / DONE: start a new study plan ───────────────────────────
            elif phase in ("idle", "done"):
                topics = run_planner(user_input)
                st.session_state.plan = topics
                st.session_state.current_step = 0
                plan_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(topics))
                bot(
                    f"📚 **Here's your study plan:**\n\n{plan_text}\n\n"
                    f"We'll start with **{topics[0]}**.\n\n"
                    "Should I go ahead and explain this topic? (yes / no)"
                )
                st.session_state.phase = "await_explain"

            elif phase == "await_explain":
                step = st.session_state.current_step
                topic = st.session_state.plan[step]
                if lowered in YES:
                    bot(run_explainer(topic))
                    st.session_state.phase = "await_quiz"
                else:
                    bot(f"Skipping explanation. Shall I quiz you on **{topic}**? (yes / no)")
                    st.session_state.phase = "await_quiz"

            elif phase == "await_quiz":
                step = st.session_state.current_step
                topic = st.session_state.plan[step]
                if lowered in YES:
                    q = run_quizzer(topic)
                    st.session_state.last_question = q
                    bot(q)
                    st.session_state.phase = "await_answer"
                else:
                    st.session_state.current_step += 1
                    advance_to_next_topic()

            elif phase == "await_answer":
                step = st.session_state.current_step
                topic = st.session_state.plan[step]
                is_correct, feedback = run_grader(
                    topic, st.session_state.last_question, user_input
                )
                bot(feedback)
                if is_correct:
                    st.session_state.current_step += 1
                    advance_to_next_topic()
                else:
                    bot("Would you like to **retry the quiz** or should I **re-explain** the topic?\n(retry / re-explain)")
                    st.session_state.phase = "await_next"

            elif phase == "await_next":
                step = st.session_state.current_step
                topic = st.session_state.plan[step] if step < len(st.session_state.plan) else None
                if lowered in ("retry", "retry quiz", "try again"):
                    q = run_quizzer(topic)
                    st.session_state.last_question = q
                    bot(q)
                    st.session_state.phase = "await_answer"
                elif lowered in ("re-explain", "explain", "re explain", "explain again"):
                    bot(run_explainer(topic))
                    st.session_state.phase = "await_quiz"
                elif lowered in YES and topic:
                    bot(f"Let's move to **{topic}**. Shall I explain it? (yes / no)")
                    st.session_state.phase = "await_explain"
                else:
                    advance_to_next_topic()

        st.rerun()

st.markdown("---")
st.caption("Built with LangChain · Streamlit | Powered by OpenAI & Wikipedia")