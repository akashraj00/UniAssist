"""Streamlit web UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
from collections import Counter

import streamlit as st

from uniassist.config import Config
from uniassist.ingest import ingest
from uniassist.loader import SUPPORTED
from uniassist.pipeline import UniAssist

st.set_page_config(page_title="UniAssist", page_icon="🎓", layout="wide")

EXAMPLES = [
    "What is the minimum attendance to write exams?",
    "How do I apply for re-evaluation and what does it cost?",
    "What are the prerequisites for Deep Learning?",
    "By what time must first-year students be in the hostel?",
    "Will I pass my exams?",  # deliberately unanswerable
]


@st.cache_resource(show_spinner="Loading index...")
def get_bot(version: int) -> UniAssist:  # `version` busts the cache after re-indexing
    return UniAssist(Config())


if "index_version" not in st.session_state:
    st.session_state.index_version = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

cfg = Config()
st.title("🎓 UniAssist")
st.caption("Source-grounded answers to university questions. It cites its sources and refuses when it does not know.")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks retrieved (k)", 1, 8, cfg.top_k)
    rerank = st.toggle("Lexical re-ranker", value=cfg.use_rerank)
    show_context = st.toggle("Show retrieved passages", value=True)

    st.header("Documents")
    st.write(f"Folder: `{cfg.docs_dir.name}/`")
    uploads = st.file_uploader("Add documents", type=[s.strip(".") for s in SUPPORTED], accept_multiple_files=True)
    if st.button("Save uploads and re-index", disabled=not uploads):
        for up in uploads:
            (cfg.docs_dir / up.name).write_bytes(up.getvalue())
        with st.spinner("Indexing..."):
            ingest(cfg)
        st.session_state.index_version += 1
        st.rerun()
    if st.button("Re-index existing documents"):
        with st.spinner("Indexing..."):
            ingest(cfg)
        st.session_state.index_version += 1
        st.rerun()

    st.header("Try these")
    for example in EXAMPLES:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.pending = example

if not (cfg.index_dir / "meta.json").exists():
    st.warning("No index yet. Click **Re-index existing documents** in the sidebar or run `python -m uniassist.ingest`.")
    st.stop()

bot = get_bot(st.session_state.index_version)
chat_tab, analytics_tab = st.tabs(["Chat", "Analytics"])

# ----------------------------------------------------------------------------- chat
with chat_tab:
    st.caption(f"Generator: **{bot.generator.name}** | Embedder: **{bot.cfg.embedder}** | Chunks indexed: **{len(bot.store)}**")

    def render(message: dict) -> None:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message["role"] == "assistant":
                for c in message.get("citations", []):
                    page = f", page {c['page']}" if c["page"] else ""
                    st.caption(f"[{c['n']}] **{c['source']}**{page} - {c['section']} (similarity {c['score']})")
                if message.get("refused"):
                    st.info("No supporting passage was found, so the assistant declined to answer.")
                if show_context and message.get("passages"):
                    with st.expander("Retrieved passages"):
                        for p in message["passages"]:
                            st.markdown(f"**#{p['rank']}** `{p['source']}` - score {p['score']:.3f}, "
                                        f"term coverage {p['overlap']:.0%}")
                            st.text(p["text"])

    for message in st.session_state.messages:
        render(message)

    prompt = st.chat_input("Ask about attendance, exams, fees, hostel, courses...")
    prompt = st.session_state.pop("pending", None) or prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "text": prompt})
        render(st.session_state.messages[-1])
        with st.spinner("Searching the documents..."):
            answer = bot.ask(prompt, k=k, rerank=rerank)
        st.session_state.messages.append({
            "role": "assistant", "text": answer.text, "citations": answer.citations, "refused": answer.refused,
            "passages": [{"rank": r.rank, "source": r.chunk.source, "score": r.score,
                          "overlap": r.overlap, "text": r.chunk.text} for r in answer.retrieved],
        })
        render(st.session_state.messages[-1])
        st.caption(f"Answered in {answer.latency_s * 1000:.0f} ms. Always confirm important decisions with the academic office.")

# ----------------------------------------------------------------------------- analytics
with analytics_tab:
    st.subheader("What are students asking?")
    if not cfg.log_path.exists():
        st.write("No questions logged yet. Ask something in the Chat tab.")
    else:
        rows = [json.loads(line) for line in cfg.log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        refused = [r for r in rows if r["refused"]]
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions asked", len(rows))
        c2.metric("Refused (not in documents)", len(refused))
        c3.metric("Answer rate", f"{100 * (1 - len(refused) / len(rows)):.0f}%")
        st.markdown("**Most-used sources**")
        sources = Counter(s for r in rows for s in r["sources"])
        if sources:
            st.bar_chart(sources)
        st.markdown("**Unanswered questions (documentation gaps)**")
        for r in refused[-15:][::-1]:
            st.write(f"- {r['question']}")
