from __future__ import annotations

import google.generativeai as genai
from typing import Iterable
import asyncio
from ..config import settings


SYSTEM_PROMPT = (
    "You are a helpful assistant. Use only the provided context to answer. "
    "If the answer is not in the context, say you don't know."
)


class LLMProvider:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        
        # Create model instance
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # For history tracking
        self.chat = self.model.start_chat(history=[])

    async def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            # Convert the message format to work with Gemini
            last_user_message = next(
                (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
                None
            )
            
            if not last_user_message:
                return "No user message found"
            
            # Since Gemini's generate_content is synchronous, run it in a thread pool
            response = await asyncio.to_thread(
                self.chat.send_message,
                last_user_message,
                generation_config={"temperature": 0.2}
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error in generate: {e}")
            raise
        # Local naive fallback: extract sentences containing query keywords
        user_last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        keywords = {w.lower() for w in user_last.split() if len(w) > 3}
        context = " ".join(m["content"] for m in messages if m["role"] == "system" and "Context:" in m["content"]) 
        sentences = context.split(". ")
        selected = [s for s in sentences if any(k in s.lower() for k in keywords)]
        if not selected:
            return "I don't know from the provided context."
        return ". ".join(selected)[:1200]

