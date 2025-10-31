from __future__ import annotations

from datetime import datetime
from typing import List, Tuple
import redis.asyncio as redis
import json
from ..config import settings
from ..schemas import ChatMessage


class ChatMemoryManager:
    def __init__(self):
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.max_turns = settings.chat_history_max_turns
        self.ttl = settings.chat_history_ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    async def add_interaction(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self._key(session_id)
        timestamp = datetime.utcnow().isoformat()
        
        # Store messages as JSON strings
        user_msg = json.dumps({
            "role": "user",
            "content": user_message,
            "timestamp": timestamp
        })
        assistant_msg = json.dumps({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": timestamp
        })

        async with self.client.pipeline(transaction=True) as pipe:
            await pipe.rpush(key, user_msg)
            await pipe.rpush(key, assistant_msg)
            await pipe.expire(key, self.ttl)
            # Trim to last N*2 messages (user+assistant pairs)
            await pipe.ltrim(key, max(0, -2 * self.max_turns), -1)
            await pipe.execute()

    async def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        key = self._key(session_id)
        items = await self.client.lrange(key, 0, -1)
        messages: List[ChatMessage] = []
        
        for item in items:
            try:
                data = json.loads(item)
                messages.append(
                    ChatMessage(
                        role=data["role"],
                        content=data["content"],
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
                
        return messages

    async def clear_history(self, session_id: str) -> None:
        key = self._key(session_id)
        await self.client.delete(key)

