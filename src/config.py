import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

def configure_llm(
    llm_model: str = "gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
):
    """Configure LlamaIndex global LLM and embedding settings."""
    Settings.llm = OpenAI(
        model=llm_model,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embed_model
    )
    print(f"⚙️  LLM configured: {llm_model} | Embed: {embed_model}")