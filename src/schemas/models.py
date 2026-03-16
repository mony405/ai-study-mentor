
"""
Step 1: The "Source of Truth" (Schemas)
Before writing any logic, you must define what a "Question," a "Response," and "Metadata" look like. This follows the Interface Segregation Principle.

File: src/schemas/models.py

Methodology: Use Pydantic.

What to do: Define class StudyMaterialMetadata, class QuizQuestion, and class MentorResponse.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone

# --- 1. Storage & Ingestion Metadata ---
class StudyMaterialMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")   # Fix 2: ignore unknown fields safely

    file_id: str
    file_name: str
    subject: str
    page_number: Optional[int] = None
    timestamp: str
    upload_date: str = ""                        # Fix 3: str not datetime (ChromaDB stores as string)
    header_path: str = ""

# --- 2. RAG & Response Components ---
class SourceCitation(BaseModel):
    """Evidence used by the AI to construct its answer."""
    text_snippet: str = Field(..., description="The actual text retrieved from the doc")
    metadata: StudyMaterialMetadata
    relevance_score: float = Field(default=0.0, description="Confidence score from the CRAG evaluator")

class MentorResponse(BaseModel):
    """The final structured object sent to the Streamlit UI."""
    answer: str = Field(..., description="The generated Markdown response")
    sources: List[SourceCitation] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence of the RAG retrieval")
    suggested_follow_up: List[str] = Field(default_factory=list, description="Contextual follow-up questions")
    
    # Metadata for the 'Efficiency Brain' and UI feedback
    is_cached: bool = Field(default=False)
    used_web_search: bool = Field(default=False)
    processing_time_ms: int = Field(default=0)

    @field_validator('answer')
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("The AI Study Mentor must provide a non-empty answer.")
        return v

    @field_validator('suggested_follow_up')
    @classmethod
    def ensure_follow_ups_on_low_confidence(cls, v: List[str], info) -> List[str]:
        # Tiered Logic: If AI is confused (<40% confident), force clarification options
        confidence = info.data.get('confidence_score', 1.0)
        if confidence < 0.4 and not v:
            return ["Could you clarify which chapter you are referring to?", "Would you like me to search the web instead?"]
        return v

# --- 3. Efficiency Brain (Cache) Schema ---
class CacheEntry(BaseModel):
    """Stored in the Semantic Cache to prevent redundant LLM calls."""
    query: str
    response: MentorResponse
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# --- 4. Interactive Quiz Module (Strict JSON) ---
class QuizOption(BaseModel):
    """A single MCQ choice."""
    label: str = Field(..., description="The label (e.g., 'A', 'B', 'C')")
    text: str = Field(..., description="The answer content text")

class QuizQuestion(BaseModel):
    question: str = Field(..., description="The MCQ question text")
    options: List[QuizOption] = Field(..., min_length=2, max_length=4)  # ✅ Pydantic v2
    correct_label: str = Field(..., description="The label (e.g., 'A') of the correct option")
    mentor_explanation: str = Field(..., description="Explanation of why the answer is correct")
    difficulty: str = Field(default="Medium", description="Easy, Medium, or Hard")

class StudyQuiz(BaseModel):
    """A collection of questions for a specific study session."""
    subject: str
    questions: List[QuizQuestion]
    total_questions: int