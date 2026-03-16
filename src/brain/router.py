import os
import asyncio
from typing import List, Any, Optional

from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Context, Event
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import ToolMetadata
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import Settings
from llama_index.tools.tavily_research import TavilyToolSpec  # type: ignore
# IMPORTANT: Ensure these match your actual folder structure!
from src.schemas.models import MentorResponse, SourceCitation, StudyMaterialMetadata, StudyQuiz
from src.modules.quiz_generator import QuizGenerator
from src.storage.index_manager import IndexFactory
from src.memory.manager import StudySessionMemory
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
import json

# --- Workflow Signals ---
class SmartDispatchEvent(Event):
    query: str
    subject: str
    action: str  # "stop" or "route"
    stop_response: Optional[str] = None

class SelectionEvent(Event):
    selected_indices: List[int]

class RetrievalEvent(Event):
    responses: List[Any]

class RelevanceEvalEvent(Event):
    relevant_nodes: List[Any]
    irrelevant_detected: bool
    confidence: Optional[float] = None

# ------prompts----------

FOLLOWUP_CHECK_PROMPT = """You are deciding if a student's message is a follow-up to the last assistant message, or a new request.

## Last assistant message:
{last_assistant_message}

## Student's new message:
{query}

---

Reply YES if the student is:
- Reacting to the last message ("thanks", "got it", "ok", "great")
- Asking to clarify, explain, or expand on something from the last message
- Using reference words that point to the last message ("that", "it", "the second point", "from the summary", "question 3")
- Asking something vague that only makes sense with the last message ("tell me more", "continue", "I don't understand")
- Sharing personal info or greeting ("hi", "my name is", "what is my name")

Reply NO if the student is:
- Asking for a quiz or practice questions
- Asking for a summary or overview of the document
- Asking a new question about a topic not mentioned in the last message
- Asking something that makes complete sense without any history

Reply with exactly YES or NO, nothing else.
"""

ROUTING_CHECK_PROMPT = """You are a scope checker for an NLP study assistant.

## Student's message:
{query}

---
Is this message related to studying, NLP, machine learning, or academics?

Reply with exactly one word:
- OUT_OF_SCOPE  → only for topics with zero academic relevance
                  such as cooking, weather, sports, personal life,
                  movies, entertainment, or general life advice
- NEEDS_ROUTING → for ANY academic or study-related question,
                  including NLP, ML, math, science, summaries,
                  quizzes, or any course-related topic
"""

FOLLOWUP_ANSWER_PROMPT = """You are a warm and encouraging study mentor.

## Conversation history:
{chat_history}

## Student's message:
{query}

---
Answer the student's message using only the conversation history above.
Be warm, concise, and encouraging. Keep your answer to 2-4 sentences.
Do not introduce new information that is not already in the history.
"""

BATCH_RELEVANCY_PROMPT: PromptTemplate = PromptTemplate(
    template="""You are evaluating document relevance for a student's question.
Question: {query_str}

For each document below, answer ONLY 'yes' or 'no' and a confidence score between 0 and 1:
{documents}

Return as JSON only, no explanation: 
{{"0": {{"relevant": "yes", "confidence": 0.9}}, 
  "1": {{"relevant": "no", "confidence": 0.2}}}}"""
)

VECTOR_PROMPT = """You are an expert AI study mentor helping a university student understand their course material.

Previous conversation:
{chat_history}

Answer the student's question using ONLY the provided course material below.

RULES:
* Be precise and focused — answer exactly what was asked
* Mention the page number where the information was found (e.g. "According to page 34...")
* Use LaTeX for all math formulas (double dollar signs $$ for blocks)
* If the answer has multiple parts, use bullet points
* Do not add information that is not in the provided material

Student question: {query}

Course material:
{context_text}"""

WEB_FALLBACK_PROMPT = """You are an expert AI study mentor helping a university student.

Previous conversation:
{chat_history}

The course material did not fully cover this topic, so additional web sources were used to answer.

Answer the student's question clearly and accurately using the provided context below.

RULES:
* Be precise and focused
* Use LaTeX for all math formulas (double dollar signs $$ for blocks)
* If the answer has multiple parts, use bullet points
* At the very end, add this note exactly:
  > ⚠️ Note: This answer was supplemented with web sources as the course material did not fully cover this topic.

Student question: {query}

Context:
{context_text}"""

SUMMARY_PROMPT = """You are a brilliant professor who has just finished reading the following study material. A student is asking you to help them understand it before an exam.

## Document: {document_title}

## Content:
{context_text}

---

Before writing anything, read the material carefully and mentally note:
- Every distinct topic or method mentioned
- Every formula, even informal ones
- Every concrete example or illustration (e.g. the bardiwac example, co-occurrence matrix numbers, one-hot vectors)
- Every limitation or trade-off mentioned

You will be required to include all of the above in your summary.

---

Now write a rich, flowing summary as if you're explaining this material to an intelligent student in office hours. Follow this structure:

**Opening.** In 2–3 sentences, tell the student what this material is fundamentally about and why it matters. What problem does it solve?

**Body — one ### section per major topic.** For each topic:
- Explain what it is in plain language
- Include the exact formula if one exists, then explain what each term means in plain language
- Use the concrete example from the material — do not replace it with a vague description
- Explain why it exists — what limitation of the previous method made it necessary
- State its own weaknesses or trade-offs
- End with a single sentence connecting it to the next topic

**Closing.** A paragraph (not a bullet list) stating the single most important mental model the student should walk away with. Ground it in the material — no generic statements about "the field."

---

## Hard rules:
- Every topic, formula, and example you noted in the pre-reading step must appear in the summary. Before finishing, scan your output against the material and add anything missing.
- ## for the document title, ### for each major topic section.
- **Bold** key terms on first use.
- Bullet points only for genuinely parallel lists (e.g. a list of limitations). No nested bullets.
- Flowing prose within each section — not bullets masquerading as paragraphs.
- Do not hedge with phrases like "it is important to note that" or "this is a dynamic field."
- Do not add information not in the material.
- Do not skip any topic to keep the summary short. Length is not a concern — completeness is.
"""

class StudyMentorWorkflow(Workflow):
    def __init__(
        self,
        index_factory: IndexFactory,
        memory: StudySessionMemory,
        tavily_key: str = None,
        timeout: int = 120,
    ):
        super().__init__(timeout=timeout)
        self.factory     = index_factory
        self.memory      = memory
        self.tavily_tool = TavilyToolSpec(api_key=tavily_key or os.getenv("TAVILY_API_KEY"))
        self.quiz_gen    = QuizGenerator(index_factory=index_factory)
 
    # ──────────────────────────────────────────────────────────────────────────
    @step
    async def initialize_session(self, ctx: Context, ev: StartEvent) -> SmartDispatchEvent:
        """Step 1: Initialize context and save user intent."""
        query   = ev.get("query")
        subject = ev.get("subject")
        mode    = ev.get("mode", "chat")
 
        print(f"\n{'='*50}")
        print(f"📥 NEW QUERY: '{query}'")
        print(f"   Subject: {subject} | Mode: {mode}")
        print(f"{'='*50}")
 
        await self.memory.memory.aput(ChatMessage(role=MessageRole.USER, content=query))
        history = await self.memory.get_active_history(mode=mode)
 
        await ctx.store.set("query",         query)
        await ctx.store.set("subject",       subject)
        await ctx.store.set("chat_history",  history)
        await ctx.store.set("mode",          mode)
        await ctx.store.set("num_questions", ev.get("num_questions", 10))
        await ctx.store.set("difficulty",    ev.get("difficulty", "Mixed"))
 
        return SmartDispatchEvent(query=query, subject=subject, action="route")
 
    # ──────────────────────────────────────────────────────────────────────────
    @step
    async def smart_dispatcher(
        self, ctx: Context, ev: SmartDispatchEvent
    ) -> SelectionEvent | StopEvent:
        """Step 2: Two-prompt dispatcher — follow-up check then scope check."""
        query   = await ctx.store.get("query")
        history = await ctx.store.get("chat_history")
        mode    = await ctx.store.get("mode")
 
        print(f"\n🧠 STEP 2 — Smart Dispatcher (2-prompt architecture)")
 
        # ── Short-circuit for explicit quiz mode ──────────────────────
        if mode == "quiz":
            print(f"   ⚡ Mode=quiz → short-circuit to quiz task")
            await ctx.store.set("task_type", 2)
            return SelectionEvent(selected_indices=[2])
 
        # ── Build full history text ───────────────────────────────────
        history_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in history
        ]) if history else "No previous conversation."
 
        print(f"   📜 History length: {len(history)} messages")
 
        # ── Extract last assistant message for Prompt A ───────────────
        last_assistant_message = "No previous message."
        if history:
            for msg in reversed(history):
                if msg.role.value in ("assistant", "ASSISTANT"):
                    last_assistant_message = msg.content
                    break
 
        print(f"   📌 Last assistant msg: '{last_assistant_message[:80]}...'")
 
        # ── PROMPT A: Is this a follow-up? ────────────────────────────
        followup_prompt = FOLLOWUP_CHECK_PROMPT.format(
            last_assistant_message=last_assistant_message,
            query=query
        )
        followup_res = await Settings.llm.acomplete(followup_prompt)
        is_followup  = followup_res.text.strip().upper()
 
        print(f"   🔍 Follow-up check result: '{is_followup}'")
 
        if is_followup == "YES":
            # Answer directly from history
            answer_prompt = FOLLOWUP_ANSWER_PROMPT.format(
                chat_history=history_text,
                query=query
            )
            answer_res = await Settings.llm.acomplete(answer_prompt)
            response   = answer_res.text.strip()
 
            print(f"   💬 Handled directly → follow-up or chitchat")
            await self.memory.memory.aput(
                ChatMessage(role=MessageRole.ASSISTANT, content=response)
            )
            return StopEvent(result=response)
 
        # ── PROMPT B: Is this in scope? ───────────────────────────────
        routing_prompt = ROUTING_CHECK_PROMPT.format(query=query)
        routing_res    = await Settings.llm.acomplete(routing_prompt)
        routing        = routing_res.text.strip().upper()
 
        print(f"   🗺️  Routing check result: '{routing}'")
 
        if "OUT_OF_SCOPE" in routing:
            print(f"   🚫 Routed → OUT OF SCOPE")
            res = "I'm focused on your course material. Let's stick to the subject!"
            await self.memory.memory.aput(
                ChatMessage(role=MessageRole.ASSISTANT, content=res)
            )
            return StopEvent(result=res)
 
        # ── Route to task selector ────────────────────────────────────
        print(f"   🔀 Needs routing → passing to task selector")
        return await self._select_task(ctx, query)
 
    # ──────────────────────────────────────────────────────────────────────────
    async def _select_task(self, ctx: Context, query: str) -> SelectionEvent:
        """Internal: select the right task type for a material query."""
        print(f"\n📋 TASK SELECTOR")
 
        selector = PydanticSingleSelector.from_defaults()
        choices  = [
            ToolMetadata(
                name="vector",
                description=(
                    "Use for ANY factual question, concept explanation, definition, or topic lookup — "
                    "regardless of whether it is in the course material or not. "
                    "This is the DEFAULT choice for all knowledge questions. "
                    "Examples: 'what is the attention mechanism?', 'explain backpropagation', "
                    "'what is the Attention Is All You Need paper?', 'how does dropout work?', "
                    "'what is a transformer?'. "
                    "When in doubt between vector and summary, always choose vector."
                ),
            ),
            ToolMetadata(
                name="summary",
                description=(
                    "Use ONLY when the student explicitly asks to summarize, outline, or get an overview "
                    "of the ENTIRE uploaded course document. "
                    "Required trigger phrases: 'summarize the document', 'summarize the PDF', "
                    "'give me an overview of the material', 'what are the main topics of this file', "
                    "'summarize what we are studying'. "
                    "Do NOT use for any question that asks about a specific concept, paper, or topic — "
                    "even if the word 'explain' or 'overview' appears in the question."
                ),
            ),
            ToolMetadata(
                name="quiz",
                description=(
                    "Use ONLY when the student explicitly asks to generate a quiz, test, exam questions, "
                    "or practice questions from the study material. "
                    "Examples: 'generate a quiz', 'make me a test', 'give me practice questions'."
                ),
            ),
        ]
 
        res          = await selector.aselect(choices, query)
        selected_idx = res.selections[0].index
        task_names   = ["vector", "summary", "quiz"]
 
        print(f"   ✅ Selected task: '{task_names[selected_idx]}' (index={selected_idx})")
 
        await ctx.store.set("task_type", selected_idx)
        return SelectionEvent(selected_indices=[selected_idx])
 
    # ──────────────────────────────────────────────────────────────────────────
    @step
    async def retrieve_from_storage(
        self, ctx: Context, ev: SelectionEvent
    ) -> RetrievalEvent | StopEvent:
        """Step 3: Retrieve based on task type."""
        query      = await ctx.store.get("query")
        subject    = await ctx.store.get("subject")
        task_type  = await ctx.store.get("task_type")
        task_names = ["vector", "summary", "quiz"]
 
        print(f"\n📦 STEP 3 — Retrieve from Storage")
        print(f"   Task type: {task_names[task_type]} ({task_type})")
 
        # Quiz path
        if task_type == 2:
            num_questions = await ctx.store.get("num_questions") or 10
            difficulty    = await ctx.store.get("difficulty") or "Mixed"
            print(f"   🎯 Generating quiz: {num_questions} questions | difficulty={difficulty}")
            quiz_obj = await self.quiz_gen.generate_quiz(
                subject=subject,
                num_questions=num_questions,
                difficulty=difficulty,
            )
            print(f"   ✅ Quiz generated: {quiz_obj.total_questions} questions")
            return StopEvent(result=quiz_obj)
 
        # Summary path
        if task_type == 1:
            print(f"   📄 Loading full text for summary...")
            full_text    = self.factory.get_full_text(subject)
            summary_node = NodeWithScore(
                node=TextNode(text=full_text, metadata={"subject": subject}),
                score=1.0,
            )
            await ctx.store.set("skip_crag", True)
            print(f"   ✅ Full text loaded: {len(full_text)} chars | skip_crag=True")
            return RetrievalEvent(responses=[summary_node])
 
        # Vector path
        print(f"   🔍 Querying ChromaDB...")
        engine   = await self.factory.get_query_engine(subject, index_type="vector")
        response = await engine.aquery(query)
        print(f"   ✅ ChromaDB returned {len(response.source_nodes)} nodes")
        return RetrievalEvent(responses=response.source_nodes)
 
    # ──────────────────────────────────────────────────────────────────────────
    @step
    async def evaluate_and_correct(
        self, ctx: Context, ev: RetrievalEvent
    ) -> RelevanceEvalEvent:
        """Step 4: CRAG — grade retrieved nodes."""
        query = await ctx.store.get("query")
 
        print(f"\n🔬 STEP 4 — CRAG Evaluation")
 
        try:
            skip_crag = await ctx.store.get("skip_crag")
        except Exception:
            skip_crag = False
        if not skip_crag:
            skip_crag = False
 
        if skip_crag:
            print(f"   ⏭️ CRAG skipped (summary path)")
            return RelevanceEvalEvent(
                relevant_nodes=ev.responses,
                irrelevant_detected=False,
                confidence=None,
            )
 
        print(f"   Grading {len(ev.responses)} nodes...")
        doc_text = "\n\n".join([
            f"[{i}]: {node.text[:300]}"
            for i, node in enumerate(ev.responses)
        ])
        prompt = BATCH_RELEVANCY_PROMPT.format(query_str=query, documents=doc_text)
        res    = await Settings.llm.acomplete(prompt)
 
        print(f"   🤖 CRAG raw response: {res.text[:200]}")
 
        try:
            clean    = res.text.strip().replace("```json", "").replace("```", "").strip()
            verdicts = json.loads(clean)
        except Exception as e:
            # ← improved: log the actual error and raw response for debugging
            print(f"   ⚠️ JSON parse failed: {e}")
            print(f"   ⚠️ Raw response was: {res.text[:300]}")
            verdicts = {
                str(i): {"relevant": "yes", "confidence": 0.5}
                for i in range(len(ev.responses))
            }
 
        relevant_nodes    = []
        confidence_scores = []
 
        for i, node in enumerate(ev.responses):
            verdict     = verdicts.get(str(i), {})
            is_relevant = verdict.get("relevant") == "yes"
            confidence  = verdict.get("confidence", 0.5)
            print(f"   Node [{i}]: relevant={is_relevant} | confidence={confidence}")
            if is_relevant:
                relevant_nodes.append(node)
                confidence_scores.append(confidence)
 
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )
        irrelevant_detected = not any(
            verdicts.get(str(i), {}).get("confidence", 0.0) >= 0.85
            for i in range(len(ev.responses))
        )
 
        print(
            f"   ✅ Relevant: {len(relevant_nodes)}/{len(ev.responses)} | "
            f"avg_confidence={avg_confidence:.2f} | "
            f"irrelevant_detected={irrelevant_detected}"
        )
 
        return RelevanceEvalEvent(
            relevant_nodes=relevant_nodes,
            irrelevant_detected=irrelevant_detected,
            confidence=avg_confidence,
        )
 
    # ──────────────────────────────────────────────────────────────────────────
    @step
    async def finalize_mentor_answer(
        self, ctx: Context, ev: RelevanceEvalEvent
    ) -> StopEvent:
        """Step 5: Synthesis & Web Fallback."""
        query      = await ctx.store.get("query")
        subject    = await ctx.store.get("subject")
        history    = await ctx.store.get("chat_history")
        task_type  = await ctx.store.get("task_type")
        start_time = asyncio.get_event_loop().time()
 
        print(f"\n✍️  STEP 5 — Finalize Answer")
 
        sources      = []
        context_text = ""
 
        for node in ev.relevant_nodes:
            context_text += f"\n{node.text}"
            try:
                meta = StudyMaterialMetadata(**node.metadata)
            except Exception:
                meta = StudyMaterialMetadata(
                    file_id="web",
                    file_name=node.metadata.get("source", "Web Search"),
                    subject=subject,
                    timestamp="",
                    upload_date="",
                )
            citation = SourceCitation(
                text_snippet=node.text[:400],
                metadata=meta,
                relevance_score=getattr(node, "score", 0.8),
            )
            sources.append(citation)
 
        # Build history text for prompts
        history_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in history
        ]) if history else "No previous conversation."
 
        used_web = False
        if ev.irrelevant_detected or not ev.relevant_nodes:
            print(f"   ⚠️ CRAG fallback triggered — fetching web results...")
            used_web    = True
            search_res  = self.tavily_tool.search(query, max_results=2)
            search_text = "\n".join([
                r.text if hasattr(r, "text") else r.get("content", "")
                for r in search_res
            ])
            context_text += f"\n\n[Web Context]:\n{search_text}"
            print(f"   🌐 Web results fetched: {len(search_res)} sources")
 
        # Select prompt based on path
        if task_type == 1:
            print(f"   📄 Using SUMMARY prompt")
            prompt = SUMMARY_PROMPT.format(
                document_title=subject.upper(),
                context_text=context_text,
            )
        elif used_web:
            print(f"   🌐 Using WEB FALLBACK prompt")
            prompt = WEB_FALLBACK_PROMPT.format(
                chat_history=history_text,
                query=query,
                context_text=context_text,
            )
        else:
            print(f"   🔍 Using VECTOR prompt")
            prompt = VECTOR_PROMPT.format(
                chat_history=history_text,
                query=query,
                context_text=context_text,
            )
 
        # Stream the response
        print(f"   ⏳ Streaming response...")
        final_answer = ""
        async for chunk in await Settings.llm.astream_complete(prompt):
            final_answer += chunk.delta
        final_answer = final_answer.strip()
 
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
 
        if ev.confidence is not None:
            final_confidence = round(ev.confidence * (0.7 if used_web else 1.0), 2)
        else:
            final_confidence = 0.0
 
        print(
            f"   ✅ Answer generated | confidence={final_confidence} | "
            f"used_web={used_web} | time={processing_time}ms"
        )
 
        response_obj = MentorResponse(
            answer=str(final_answer),
            sources=sources,
            confidence_score=final_confidence,
            is_cached=False,
            used_web_search=used_web,
            processing_time_ms=processing_time,
        )
 
        await self.memory.memory.aput(
            ChatMessage(role=MessageRole.ASSISTANT, content=response_obj.answer)
        )
        return StopEvent(result=response_obj)



# ---------------testing----------------------

import asyncio
import os
from dotenv import load_dotenv
import nest_asyncio

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from src.storage.index_manager import IndexFactory
from src.memory.manager import StudySessionMemory
from src.brain.router import StudyMentorWorkflow
from src.schemas.models import MentorResponse, StudyQuiz
nest_asyncio.apply()
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────

# ── ANSI colour helpers ───────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}{CYAN}   {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}")

def scenario(number: int, label: str, expected_path: str) -> None:
    print(f"\n{BOLD}{YELLOW}── Scenario {number:02d} │ {label}{RESET}")
    print(f"   Expected path : {expected_path}")

def result(response: str) -> None:
    preview = str(response)[:300].replace("\n", " ")
    print(f"   Response      : {GREEN}{preview}{'...' if len(str(response)) > 300 else ''}{RESET}")

# ─────────────────────────────────────────────────────────────────────────────

async def test_mentor_brain():

    header("AI STUDY MENTOR — FULL SCENARIO COVERAGE TEST")

    # ── Setup ─────────────────────────────────────────────────────────────────
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    factory  = IndexFactory(storage_dir="./test_storage_chroma")
    memory   = StudySessionMemory(session_id="test_full_coverage")
    workflow = StudyMentorWorkflow(
        index_factory=factory,
        memory=memory,
        tavily_key=os.getenv("TAVILY_API_KEY"),
    )
    subject = "nlp"

    # # =========================================================================
    # # BLOCK A — DISPATCHER SELF-HANDLED (no routing needed)
    # # =========================================================================

    # # ── A1: Greeting / chitchat ───────────────────────────────────────────────
    # scenario(1, "Greeting — dispatcher should answer directly", "DISPATCHER → direct answer")
    # r = await workflow.run(query="Hi! I am Menna and I need help studying NLP.", subject=subject)
    # result(r)

    # # ── A2: Personal fact from history (name recall) ──────────────────────────
    # scenario(2, "Name recall — history contains the name", "DISPATCHER → direct answer from history")
    # r = await workflow.run(query="What is my name?", subject=subject)
    # result(r)

    # # ── A3: Simple thanks / filler ────────────────────────────────────────────
    # scenario(3, "Chitchat thanks — dispatcher should answer directly", "DISPATCHER → direct answer")
    # r = await workflow.run(query="Thanks, that was really helpful!", subject=subject)
    # result(r)

    # # =========================================================================
    # # BLOCK B — VECTOR PATH (ChromaDB hit — topic IS in the PDF)
    # # =========================================================================

    # # ── B1: Direct concept question ───────────────────────────────────────────
    # scenario(4, "Concept in PDF — cosine similarity", "NEEDS_ROUTING → vector → ChromaDB hit")
    # r = await workflow.run(
    #     query="How can we compare the similarity between words in the embedding space?",
    #     subject=subject,
    # )
    # result(r)

    # # ── B2: Formula / math question ───────────────────────────────────────────
    # scenario(5, "Formula in PDF — PMI formula", "NEEDS_ROUTING → vector → ChromaDB hit")
    # r = await workflow.run(query="What is the formula for Pointwise Mutual Information?", subject=subject)
    # result(r)

    # # ── B3: Definition lookup ─────────────────────────────────────────────────
    # scenario(6, "Definition in PDF — TF-IDF", "NEEDS_ROUTING → vector → ChromaDB hit")
    # r = await workflow.run(query="What is TF-IDF and why is it useful?", subject=subject)
    # result(r)

    # # ── B4: Comparison question ───────────────────────────────────────────────
    # scenario(7, "Comparison in PDF — Word2Vec vs PMI", "NEEDS_ROUTING → vector → ChromaDB hit")
    # r = await workflow.run(
    #     query="What is the relationship between Word2Vec skip-gram and PMI?",
    #     subject=subject,
    # )
    # result(r)

    # # ── B5: List/enumeration question ────────────────────────────────────────
    # scenario(8, "List question in PDF — problems with one-hot", "NEEDS_ROUTING → vector → ChromaDB hit")
    # r = await workflow.run(query="What are the problems with one-hot word representations?", subject=subject)
    # result(r)

    # # ── B6: Follow-up on a VECTOR answer already in history ───────────────────
    # scenario(9, "Follow-up on vector answer — dispatcher should self-answer", "DISPATCHER → direct answer from history")
    # r = await workflow.run(query="Can you elaborate more on the second problem you just mentioned?", subject=subject)
    # result(r)

    # # ── B7: Clarification request on prior answer ─────────────────────────────
    # scenario(10, "Clarification on prior answer — 'what did you mean by'", "DISPATCHER → direct answer from history")
    # r = await workflow.run(query="What did you mean by polysemy just now?", subject=subject)
    # result(r)

    # # =========================================================================
    # # BLOCK C — WEB FALLBACK PATH (topic NOT in PDF → CRAG → Tavily)
    # # =========================================================================

    # # ── C1: Paper not in PDF ──────────────────────────────────────────────────
    # scenario(11, "Topic NOT in PDF — Attention Is All You Need paper", "NEEDS_ROUTING → vector → CRAG fallback → web")
    # r = await workflow.run(
    #     query="What does 'Attention Is All You Need' mean in the world of NLP?",
    #     subject=subject,
    # )
    # result(r)

    # # ── C2: Recent/external concept ───────────────────────────────────────────
    # scenario(12, "Topic NOT in PDF — BERT architecture", "NEEDS_ROUTING → vector → CRAG fallback → web")
    # r = await workflow.run(query="Can you explain how BERT works?", subject=subject)
    # result(r)

    # # ── C3: Follow-up on a WEB FALLBACK answer ────────────────────────────────
    # scenario(13, "Follow-up on web answer — dispatcher self-answers from history", "DISPATCHER → direct answer from history")
    # r = await workflow.run(query="I don't understand the last point you explained, can you simplify it?", subject=subject)
    # result(r)

    # # =========================================================================
    # # BLOCK D — SUMMARY PATH
    # # =========================================================================

    # # ── D1: Explicit summary request ─────────────────────────────────────────
    # scenario(14, "Explicit summary request", "NEEDS_ROUTING → summary path")
    # r = await workflow.run(query="Can you give me a full summary of the document?", subject=subject)
    # result(r)

    # # ── D2: Follow-up on summary point — dispatcher must NOT re-summarize ─────
    # scenario(15, "Follow-up on summary — 'explain point X' — NO re-routing", "DISPATCHER → direct answer from history")
    # r = await workflow.run(
    #     query="I don't understand the last point from the summary. What does it mean?",
    #     subject=subject,
    # )
    # result(r)

    # # ── D3: Specific item from summary ───────────────────────────────────────
    # scenario(16, "Ask about specific item mentioned in summary", "DISPATCHER → direct answer from history")
    # r = await workflow.run(query="Can you elaborate on the Bag-of-Words section from the summary?", subject=subject)
    # result(r)

    # # ── D4: Second summary request — should re-summarize (new request) ────────
    # scenario(17, "Second summary request — different phrasing", "NEEDS_ROUTING → summary path")
    # r = await workflow.run(query="Give me an outline of the main topics covered.", subject=subject)
    # result(r)

    # # =========================================================================
    # # BLOCK E — QUIZ PATH
    # # =========================================================================

    # # ── E1: Basic quiz request ────────────────────────────────────────────────
    # scenario(18, "Quiz generation request — 5 questions", "NEEDS_ROUTING → quiz path")
    # r = await workflow.run(query="Make me a 5-question quiz on this material.", subject=subject)
    # result(r)

    # # ── E2: Follow-up on quiz question ───────────────────────────────────────
    # scenario(19, "Follow-up on quiz — ask about a specific question", "DISPATCHER → direct answer from history")
    # r = await workflow.run(query="I don't know the answer to question 3. Can you explain it?", subject=subject)
    # result(r)

    # ── E3: New quiz request after one exists (should re-route to quiz) ────────
    scenario(20, "New quiz request — harder difficulty", "NEEDS_ROUTING → quiz path")
    r = await workflow.run(query="Now give me a harder quiz with 3 questions about Word2Vec only.", subject=subject)
    result(r)

    # =========================================================================
    # BLOCK F — OUT OF SCOPE
    # =========================================================================

    # ── F1: Completely unrelated topic ───────────────────────────────────────
    scenario(21, "Out of scope — cooking recipe", "DISPATCHER → OUT_OF_SCOPE")
    r = await workflow.run(query="How do I make pasta carbonara?", subject=subject)
    result(r)

    # ── F2: Unrelated but sounds academic ────────────────────────────────────
    scenario(22, "Out of scope — history question", "DISPATCHER → OUT_OF_SCOPE")
    r = await workflow.run(query="Who won World War II?", subject=subject)
    result(r)

    # ── F3: Borderline — general AI but not NLP material ─────────────────────
    scenario(23, "Borderline — general ML question (should route to vector/web)", "NEEDS_ROUTING → vector or web fallback")
    r = await workflow.run(query="What is gradient descent?", subject=subject)
    result(r)

    # =========================================================================
    # BLOCK G — EDGE CASES
    # =========================================================================

    # ── G1: Empty-history follow-up attempt (new session) ────────────────────
    scenario(24, "Edge case — clarification with empty history", "NEEDS_ROUTING → vector (history is empty)")
    fresh_memory   = StudySessionMemory(session_id="test_empty_history")
    fresh_workflow = StudyMentorWorkflow(
        index_factory=factory,
        memory=fresh_memory,
        tavily_key=os.getenv("TAVILY_API_KEY"),
    )
    r = await fresh_workflow.run(query="Can you clarify what you said about embeddings?", subject=subject)
    result(r)

    # ── G2: Ambiguous — "explain" keyword but new topic ──────────────────────
    scenario(25, "Edge case — 'explain' keyword but topic not in history", "NEEDS_ROUTING → vector")
    r = await workflow.run(query="Explain negative sampling to me.", subject=subject)
    result(r)

    # ── G3: Repeated exact question — should be answered from history ─────────
    scenario(26, "Edge case — repeated exact question already answered", "DISPATCHER → direct answer from history")
    r = await workflow.run(query="What is TF-IDF and why is it useful?", subject=subject)
    result(r)

    # ── G4: Multi-part question spanning PDF and web ──────────────────────────
    scenario(27, "Edge case — multi-part: one part in PDF, one outside", "NEEDS_ROUTING → vector (may trigger web fallback for second part)")
    r = await workflow.run(
        query="What is Word2Vec and how does it compare to GPT embeddings?",
        subject=subject,
    )
    result(r)

    # ── G5: Vague query with no clear intent ─────────────────────────────────
    scenario(28, "Edge case — vague query", "DISPATCHER → clarification / direct answer")
    r = await workflow.run(query="Tell me more.", subject=subject)
    result(r)

    # ── G6: Math formula request (should hit vector, expect LaTeX output) ─────
    scenario(29, "Edge case — math formula expected in LaTeX", "NEEDS_ROUTING → vector → LaTeX in response")
    r = await workflow.run(
        query="Write the softmax probability formula used in Word2Vec skip-gram.",
        subject=subject,
    )
    result(r)

    # ── G7: Summary then immediate unrelated question ─────────────────────────
    scenario(30, "Edge case — unrelated question right after summary", "DISPATCHER → OUT_OF_SCOPE")
    r = await workflow.run(query="What is the weather like today?", subject=subject)
    result(r)

    # =========================================================================
    print(f"\n{BOLD}{GREEN}{'═' * 65}{RESET}")
    print(f"{BOLD}{GREEN}   ALL 30 SCENARIOS COMPLETED{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 65}{RESET}\n")


if __name__ == "__main__":
    asyncio.run(test_mentor_brain())