"""
Step 6: The "Interaction Layer" (Memory & UI)
Finally, wrap it in a session.

File: src/memory/manager.py

Class: StudySessionMemory

File: app/main_ui.py

Action: Build the Streamlit interface that calls the TieredRouter.
"""
import os
from llama_index.core.memory import (
    Memory, 
    FactExtractionMemoryBlock,
    VectorMemoryBlock
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from typing import List, Optional

class StudySessionMemory:
    def __init__(self, session_id: str, token_limit: int = 4000):
        """
        Initializes the modern LlamaIndex Memory (2026).
        Uses an in-memory SQLite database by default.
        """
        # 1. Define Long-Term Memory Blocks
        # This extracts 'facts' from old messages before they are flushed
        blocks = [
            FactExtractionMemoryBlock(
                name="student_facts",
                llm=Settings.llm,
                max_facts=20,
                priority=0  # High priority: never truncate student facts
            )
        ]

        # 2. Initialize the modern Memory class
        # This replaces the deprecated ChatMemoryBuffer
        self.memory = Memory.from_defaults(
            session_id=session_id,
            token_limit=token_limit,
            memory_blocks=blocks,
            chat_history_token_ratio=0.7,
            insert_method="system" # Inserts long-term facts into the system prompt
        )

    async def get_active_history(self, mode: str = "chat") -> List[ChatMessage]:
        """
        Retrieves combined Short-Term (FIFO) and Long-Term (Facts) memory.
        """
        if mode in ["quiz", "summary"]:
            return []
        
        # get() returns the merged chat history ready for the LLM
        return self.memory.get()

    async def add_interaction(self, user_query: str, ai_response: str):
        """
        Stores a new interaction. 
        If token_limit is hit, old messages are 'flushed' into the FactExtractionBlock.
        """
        self.memory.put_messages([
            ChatMessage(role=MessageRole.USER, content=user_query),
            ChatMessage(role=MessageRole.ASSISTANT, content=ai_response)
        ])

    def reset(self):
        """Wipes the session history from the database."""
        self.memory.reset()