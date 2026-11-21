
from typing import Any

from langchain_core.messages import BaseMessage

from ai_engine.llm.openai_client import create_chat_model


class LLMGateway:
    """
    Central interface for all LLM interactions in the application.
    """

    def __init__(self):
        self.model = create_chat_model()

    def invoke(
        self,
        messages: Any,
        **kwargs
    ) -> BaseMessage:
        """
        Sends a request to the configured LLM.
        Returns the model response.
        """
        return self.model.invoke(messages, **kwargs)

    def invoke_text(self, prompt: str) -> str:
        """
        Sends a text prompt and returns the generated text.
        """
        response = self.invoke(prompt)

        if not isinstance(response.content, str):
            raise TypeError("Expected a text response from the LLM.")

        return response.content

    def invoke_structured(self, prompt: str, schema: Any) -> Any:
        """
        Requests output matching a supplied Pydantic schema
        or supported structured-output schema.
        """
        structured_model = self.model.with_structured_output(schema)

        return structured_model.invoke(prompt)
