# 🎓 UniAssist: a RAG-based academic helpdesk assistant

A chatbot that answers student questions about university rules, courses, fees, hostel, exams and more.
Every answer is **grounded in official documents**, **cites its sources**, and the bot **refuses** when
the documents do not contain the answer.

It runs fully **offline** out of the box (no API key, no model download). You can upgrade it to dense
embeddings and a real LLM when you want better paraphrase handling and fluent answers.

---

## 1. Run it in VS Code (5 steps)

1. **Open the folder:** *File → Open Folder…* and choose `uniassist_project`. Accept the prompt to install the recommended Python extension.
2. **Create the environment:** *Terminal → Run Task… → "Setup: create venv and install"*.
   (Manual alternative: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.)
3. **Select the interpreter:** `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac) → *Python: Select Interpreter* → choose the one inside `.venv`.
4. **Build the index:** open *Run and Debug* (`Ctrl+Shift+D`), pick **"1. Build index (ingest)"**, press `F5`.
5. **Start the app:** pick **"3. Streamlit web app"**, press `F5`, then open <http://localhost:8501>.

Other launch configurations: **2. Chat in terminal**, **4. Evaluate + ablations**, **5. Run tests**.
Or from any terminal: `python -m uniassist.ingest`, `streamlit run app.py`, `python -m uniassist.evaluate`, `pytest -q`.

> The 9 documents in `data/documents/` are **fictional** ("Example Institute of Technology"). To use your own university,
> replace them with your real `.md`, `.txt` or `.pdf` files (or upload in the web app) and re-index.

---

## 2. How it works

```
OFFLINE (ingest)                                   ONLINE (each question)
documents (.md/.txt/.pdf)                          question
   │ loader.py                                        │ guardrails.py   -> refuse predictions / injection / off-topic tasks
   ▼                                                  │ prompts.py      -> sanitise input
heading-aware chunks (+ overlap)                      ▼
   │ chunker.py                                    embed query ──► vector search (cosine, top-k)
   ▼                                                  │ retriever.py    -> lexical re-rank (optional)
embeddings (TF-IDF, or MiniLM)                        ▼
   │ embeddings.py                                 confidence gate: best score + word coverage too low? -> REFUSE
   ▼                                                  │ pipeline.py
vector store on disk                                  ▼
   vector_store.py                                 generator (extractive offline, or an LLM) with [n] citations
                                                      │ llm.py
                                                      ▼
                                                   answer + sources   (NOT_FOUND from the model -> REFUSE)
```

| Course concept | Where it lives |
|---|---|
| Chunking with overlap | `uniassist/chunker.py` |
| Embeddings, cosine similarity | `uniassist/embeddings.py`, `uniassist/vector_store.py` |
| Retrieval, re-ranking | `uniassist/retriever.py` |
| Prompt engineering, grounding, citations | `uniassist/prompts.py` |
| LLM integration (Anthropic / OpenAI / Ollama) | `uniassist/llm.py` |
| Hallucination control (refusal gate, `NOT_FOUND`) | `uniassist/pipeline.py` |
| Safety guardrails, prompt-injection handling | `uniassist/guardrails.py`, `prompts.py` |
| Evaluation: Recall@k, MRR, faithfulness, ablations | `uniassist/evaluate.py` |
| UI + usage analytics | `app.py` |

Key formulas used in the code: cosine similarity `q·d / (‖q‖‖d‖)`, `MRR = (1/|Q|) Σ 1/rank_i`,
and `Recall@k` = share of questions whose gold passage appears in the top k.

---

## 3. Results (measured, offline mode: TF-IDF + extractive answers)

Two question sets are included, and the difference between them matters:

* `eval/questions.jsonl` (54 questions): I **adjusted the system while looking at these** (fixing the answer selector and the
  refusal gate), so these numbers are optimistic.
* `eval/heldout.jsonl` (18 questions): written afterwards, deliberately **paraphrased**, and run **once** with no further tuning.
  This is the number to trust.

| Metric | Tuned set (54 Q) | **Held-out (18 Q)** |
|---|---|---|
| Retrieval Recall@4 | 1.00 | **0.92** |
| Retrieval MRR | 1.00 | **0.65** |
| Answer accuracy (expected facts present) | 97.7% | **66.7%** |
| False refusals (answerable but refused) | 0% | **16.7%** |
| Out-of-scope questions refused | 100% | **83.3%** |

**Ablations on the held-out set** (12 answerable questions, so treat differences of one or two questions as noise):

| Chunk size (words) | 40 | 80 | 120 | 240 |
|---|---|---|---|---|
| MRR | 0.59 | 0.61 | 0.65 | 0.65 |

| k (chunks retrieved) | 1 | 3 | 5 | 8 |
|---|---|---|---|---|
| Recall@k | 0.42 | 0.92 | 0.92 | 0.92 |

Re-ranker: MRR 0.72 **without** vs 0.65 **with** on the held-out set (the reverse of the tuned set, where it helped by 0.02).
The honest conclusion is that it is **not clearly helpful** here. Re-run the ablation on your own documents before relying on it.

Regenerate everything with `python -m uniassist.evaluate` and `python -m uniassist.evaluate --questions eval/heldout.jsonl`.
Reports are written to `eval/results*.md`.

---

## 4. Known limitations (read before presenting)

* **Offline mode is lexical.** It matches *words*, not meaning. "Senior student" does not match "all other students";
  "penalty" does not match "score of zero". That is the main cause of the held-out drop. Switch on MiniLM and/or an LLM (section 5).
* **The MiniLM and LLM backends are written but were NOT run while building this project.** The build sandbox had no access to
  the model host and no API key. The offline path is what the tests and numbers above cover. Test the others yourself.
* **Faithfulness is a lexical proxy.** It checks that answer words appear in the retrieved text. For the extractive generator it is
  ~1.0 by construction, so it tells you little there. For an LLM it is more meaningful, but an LLM judge would be stricter.
* **The refusal gate and guardrails are heuristics.** The thresholds (`min_score`, `min_overlap`) were set for the sample documents.
  Re-tune them on your own corpus. The guardrail patterns are narrow rules, not a classifier.
* **Small, self-authored evaluation.** The documents, questions and system were all written by the same author, and the sets are tiny.
  For a stronger report, collect real student questions and have a second person label them.
* **Sample documents are fictional.** Do not present their rules as a real university's policy.

---

## 5. Optional upgrades

```bash
pip install -r requirements-optional.txt   # sentence-transformers, anthropic, openai
cp .env.example .env                        # then edit
```

| Goal | Setting in `.env` |
|---|---|
| Dense semantic retrieval | `UNIASSIST_EMBEDDER=minilm`, then re-run `python -m uniassist.ingest` |
| Claude answers | `UNIASSIST_GENERATOR=anthropic` and `ANTHROPIC_API_KEY=...` |
| OpenAI answers | `UNIASSIST_GENERATOR=openai` and `OPENAI_API_KEY=...` |
| Local open model | install Ollama, `ollama pull llama3.2`, `UNIASSIST_GENERATOR=ollama` |

With `minilm`, re-run the evaluation and **re-tune** `UNIASSIST_MIN_SCORE` (dense cosine scores sit on a different scale; the
default of 0.30 is a starting guess). With an LLM backend, you can also run the **"grounded vs plain prompt"** ablation:
`bot.ask(question, grounded=False)` calls the model with no retrieval, which is the clearest way to show hallucination reduction
in your report.

---

## 6. Project layout

```
app.py                     Streamlit UI (chat, retrieved passages, document upload, analytics tab)
uniassist/                 the package (see the table in section 2)
data/documents/            9 sample university documents (replace with real ones)
eval/questions.jsonl       54 tuned-on questions      eval/heldout.jsonl   18 held-out questions
eval/results*.md           generated reports
tests/test_uniassist.py    23 tests (chunker, metrics, guardrails, end-to-end answers and refusals)
.vscode/                   launch configs, tasks, settings
```

## 7. Ideas for extensions

Multilingual questions (Hindi, Tamil, Kannada) with a multilingual embedder; a voice interface; quiz/summary generation from course
material; use the Analytics tab's "unanswered questions" list to find gaps in the university's documentation.
