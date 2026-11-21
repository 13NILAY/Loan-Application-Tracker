
import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI


def create_chat_model() -> ChatOpenAI:
    """
    Creates the centrally configured OpenAI-compatible chat model.
    """

    load_dotenv(find_dotenv())

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    if not model_name:
        raise ValueError("OPENAI_MODEL is missing.")

    if not base_url:
        raise ValueError("OPENAI_BASE_URL is missing.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=60,
        max_retries=2,
    )
