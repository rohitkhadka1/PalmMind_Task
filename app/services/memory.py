from __future__ import annotations

from datetime import datetime
from typing import List
import redis.asyncio as redis
import json
from ..config import settings
from ..schemas import ChatMessage


class ChatMemoryManager:
    def __init__(self):
        try:
            self.client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True
            )
            self.max_turns = settings.chat_history_max_turns
            self.ttl = settings.chat_history_ttl_seconds
        except Exception as e:
            print(f"Failed to initialize Redis client: {e}")
            raise

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    async def add_interaction(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self._key(session_id)
        timestamp = datetime.utcnow().isoformat()
        
        try:
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
        except redis.RedisError as e:
            print(f"Redis error in add_interaction: {e}")
            raise

    async def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        key = self._key(session_id)
        messages: List[ChatMessage] = []
        
        try:
            items = await self.client.lrange(key, 0, -1)
            
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
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing message: {e}")
                    continue
                    
        except redis.RedisError as e:
            print(f"Redis error in get_chat_history: {e}")
            
        return messages

    async def clear_history(self, session_id: str) -> None:
        key = self._key(session_id)
        try:
            await self.client.delete(key)
        except redis.RedisError as e:
            print(f"Redis error in clear_history: {e}")
            raise

