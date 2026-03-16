# 📖 AI Study Mentor

> An intelligent, RAG-powered study assistant that helps university students deeply understand their course material through conversational Q&A, auto-generated summaries, and adaptive quizzes.

---

## ✨ Features

- **Conversational Q&A** — Ask any question about your uploaded PDF and get precise, cited answers grounded in your course material
- **Smart Document Summaries** — Generate rich, professor-quality summaries covering every topic, formula, and example in the material
- **Adaptive Quizzes** — Auto-generate multiple-choice quizzes with configurable difficulty (Easy / Medium / Hard / Mixed) and instant feedback
- **CRAG Web Fallback** — When your course material doesn't cover a topic, the system automatically falls back to web search (via Tavily) for supplemental context
- **Persistent Session Memory** — The assistant remembers facts about you and your prior questions across a session using LlamaIndex's modern `Memory` system with `FactExtractionMemoryBlock`
- **Semantic Cache** — Repeated or semantically similar questions are served instantly without redundant LLM calls
- **LaTeX Rendering** — Math formulas are rendered properly in the Streamlit UI
- **Source Citations** — Every answer includes the page number and source snippet it was retrieved from

---

## 🏗️ Architecture

```
User (Streamlit UI)
        │
        ▼
┌─────────────────────────────────────┐
│         StudyMentorWorkflow         │  ← LlamaIndex Workflow (async)
│  (src/brain/router.py)              │
│                                     │
│  Step 1: initialize_session         │
│  Step 2: smart_dispatch             │  ← Follow-up check + scope filter
│  Step 3: retrieve_documents         │  ← ChromaDB vector retrieval
│  Step 4: evaluate_relevance         │  ← CRAG batch relevance scoring
│  Step 5: generate_answer            │  ← LLM generation + web fallback
└─────────────────────────────────────┘
        │                   │
        ▼                   ▼
  ChromaDB              Tavily Search
  (Vector Store)        (Web Fallback)
        │
        ▼
  LlamaParse → TextNodes → IndexFactory
  (PDF Ingestion)        (src/storage/)
```

### Key Components

| Module | Path | Responsibility |
|---|---|---|
| **Router / Workflow** | `src/brain/router.py` | Orchestrates the full RAG pipeline as a multi-step async workflow |
| **Document Processor** | `src/ingestion/processor.py` | Parses PDFs with LlamaParse into page-level TextNodes with rich metadata |
| **Index Factory** | `src/storage/index_manager.py` | Manages ChromaDB vector collections per subject |
| **Memory Manager** | `src/memory/manager.py` | Session memory with short-term FIFO + long-term fact extraction |
| **Quiz Generator** | `src/modules/quiz_generator.py` | Generates structured MCQ quizzes from full document text |
| **Semantic Cache** | `src/utils/cache.py` | Caches prior answers using vector similarity (threshold: 0.92) |
| **Schemas** | `src/schemas/models.py` | Pydantic models for all data contracts |
| **UI** | `main.py` | Streamlit interface with editorial dark design |

---

## 🧠 How It Works

### 1. Document Ingestion
PDFs are parsed using **LlamaParse** in `agentic` mode, which converts them to Markdown, handles math as LaTeX, converts diagrams to Mermaid.js, and preserves tables. Each page becomes a `TextNode` with structured metadata (`file_name`, `subject`, `page_number`, `upload_date`).

### 2. Indexing
Nodes are embedded with `text-embedding-3-small` and stored in **ChromaDB**, organized into per-subject collections (e.g., `nlp_collection`). The full text of each document is also saved as a `.txt` file for use in quiz and summary generation.

### 3. Routing (Smart Dispatch)
Every incoming query goes through a two-stage classifier:
- **Follow-up check** — If the query is a reaction to the last assistant message, it's answered directly from session history (no retrieval needed)
- **Scope check** — Out-of-scope queries (cooking, sports, etc.) are politely declined; academic queries proceed to retrieval

### 4. CRAG Retrieval
Relevant nodes are fetched from ChromaDB, then scored for relevance using a **batch LLM evaluator** (CRAG pattern). If too many nodes are irrelevant or no nodes are found, the system falls back to **Tavily web search** to supplement context.

### 5. Answer Generation
The LLM generates a response using one of three prompt strategies:
- **Vector prompt** — Grounded in retrieved course material with page citations
- **Summary prompt** — Comprehensive professor-style summary of the full document
- **Web fallback prompt** — Supplemented with web context, flagged with a ⚠️ note

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key
- A LlamaCloud API key (for LlamaParse)
- A Tavily API key (for web fallback search)

### Installation

```bash
git clone https://github.com/mony405/ai-study-mentor.git
cd ai-study-mentor
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Running the App

```bash
streamlit run main.py
```

The app will open in your browser. Use the sidebar to:
1. Enter a subject name (e.g., "NLP", "Biology Chapter 3")
2. Upload a PDF of your study material
3. Click **Process Document**
4. Start asking questions in the chat

---

## 🗂️ Project Structure

```
ai-study-mentor/
├── main.py                        # Streamlit UI entry point
├── requirements.txt
├── src/
│   ├── config.py                  # LLM/embedding configuration
│   ├── brain/
│   │   └── router.py              # Core workflow + all prompts
│   ├── ingestion/
│   │   └── processor.py           # PDF parsing with LlamaParse
│   ├── memory/
│   │   └── manager.py             # Session memory (short + long-term)
│   ├── modules/
│   │   └── quiz_generator.py      # MCQ quiz generation
│   ├── schemas/
│   │   └── models.py              # Pydantic data models
│   ├── storage/
│   │   └── index_manager.py       # ChromaDB index management
│   └── utils/
│       └── cache.py               # Semantic response cache
└── test_storage_chroma/           # Dev-mode pre-indexed NLP material
    └── full_text/
        └── nlp.txt
```

---

## 🧪 Testing

The project includes inline test suites for each major component. Run them individually:

```bash
# Test the document ingestion pipeline
python -m src.ingestion.processor

# Test the ChromaDB index factory
python -m src.storage.index_manager

# Test the full workflow with 30 scenarios
python -m src.brain.router
```

The workflow test covers 30 scenarios across 7 categories: greetings/chitchat, vector retrieval, web fallback, quiz generation, summary generation, out-of-scope rejection, and edge cases.

---

## ⚙️ Configuration

### Switching LLM Models

In `main.py` and test files, the LLM is configured via LlamaIndex `Settings`:

```python
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

Replace `"gpt-4o-mini"` with any OpenAI-compatible model (e.g., `"gpt-4o"`, `"gpt-4.1"`).

### Dev Mode

`main.py` includes a `DEV_MODE` flag that auto-loads the pre-indexed NLP material from `./test_storage_chroma` so you can test without uploading a PDF:

```python
DEV_MODE    = True
DEV_SUBJECT = "nlp"
DEV_STORAGE = "./test_storage_chroma"
```

Set `DEV_MODE = False` for production use.

---

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `llama-index` | RAG framework, workflows, memory |
| `llama-parse` | Agentic PDF parsing |
| `chromadb` | Vector database |
| `openai` | LLM + embeddings |
| `tavily-python` | Web search fallback |
| `pydantic` | Data validation and schemas |
| `nest-asyncio` | Async support in Streamlit |

---

## 🗺️ Roadmap

This project is actively being developed. Here's what's planned next:

### 🔧 In Progress / Planned by Author

- **Human-in-the-Loop Review** — Allow the student or instructor to flag incorrect answers, correct them, and feed that feedback back into the system to improve future responses (the `IndexFactory` already has a commented-out hook for this)
- **Folder / Batch Ingestion** — Process an entire folder of PDFs at once, so a student can upload a full semester's worth of lecture slides and query across all of them in one session
- **More Task Types** — Expand beyond Q&A, summaries, and quizzes — e.g., flashcard generation, concept mapping, essay outline drafting, or exam question prediction

### 💡 Ideas for Future Improvements

- **Multi-Subject Cross-Search** — Query across multiple subjects simultaneously (e.g., "How does TF-IDF in NLP relate to what I learned in my Statistics course?")
- **Spaced Repetition Scheduler** — Track which quiz questions the student got wrong and re-surface them at optimal intervals using a spaced repetition algorithm (like Anki)
- **Difficulty Progression** — Automatically adjust quiz difficulty over time based on the student's performance history
- **Concept Graph** — Visualize the relationships between key concepts in the uploaded material as an interactive knowledge graph
- **Voice Input / Output** — Let students ask questions by voice and receive spoken answers, useful for studying on the go
- **Study Session Analytics** — A dashboard showing the student's progress over time: topics covered, quiz scores, weak areas, and time spent
- **Multilingual Support** — Parse and answer questions in languages other than English, useful for students studying in non-English universities
- **Collaborative Study Rooms** — Let multiple students share the same indexed material and see each other's questions and the mentor's answers in real time
- **Export Study Pack** — Generate a downloadable study pack (PDF or Word doc) combining the summary, key formulas, and a practice quiz for offline revision

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Menna Samir** — [github.com/mony405](https://github.com/mony405)
