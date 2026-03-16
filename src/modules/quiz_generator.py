from typing import List
from llama_index.core import Settings
from llama_index.core.program import LLMTextCompletionProgram
from src.schemas.models import StudyQuiz, QuizQuestion
from src.storage.index_manager import IndexFactory

QUIZ_PROMPT_TEMPLATE = """You are an expert academic quiz generator.

Create a quiz from the following study material.

Steps:
1. Identify the main sections and key concepts.
2. Generate questions that cover the entire document evenly.

Requirements:
* Generate exactly {num_questions} multiple-choice questions
* Difficulty level: {difficulty}
* If difficulty is "Mixed" — distribute questions as: 30% Easy, 40% Medium, 30% Hard
* Mix of conceptual and factual questions
* Avoid repetition
* Each question must have 3-4 options
* Include the correct answer and a short explanation for why it is correct

Document:
{context_str}

Return the output strictly in the requested JSON format."""

class QuizGenerator:
    def __init__(self, index_factory: IndexFactory):
        self.factory = index_factory

    async def generate_quiz(self, subject: str, num_questions: int = 10, difficulty: str = "Mixed") -> StudyQuiz:
        context_str = self.factory.get_full_text(subject)

        program = LLMTextCompletionProgram.from_defaults(
            output_cls=StudyQuiz,
            prompt_template_str=QUIZ_PROMPT_TEMPLATE,
            llm=Settings.llm
        )

        quiz_output = await program.acall(
            subject=subject,
            num_questions=num_questions,
            difficulty=difficulty,
            context_str=context_str
        )

        quiz_output.subject = subject
        return quiz_output