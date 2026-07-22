"""Redis storage for replica-safe pending chat requests and thread ownership."""

from __future__ import annotations

from redis.asyncio import Redis


_CLAIM_THREAD = """
local owner = redis.call('GET', KEYS[1])
if not owner then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return 1
end
if owner == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""


class RedisRequestStore:
    def __init__(self, client: Redis, request_ttl: int, ownership_ttl: int) -> None:
        self.client = client
        self.request_ttl = request_ttl
        self.ownership_ttl = ownership_ttl

    async def claim_thread(self, thread_id: str, session_id: str) -> bool:
        return bool(await self.client.eval(
            _CLAIM_THREAD,
            1,
            f"opcenter:chat:thread-owner:{thread_id}",
            session_id,
            self.ownership_ttl,
        ))

    async def save_request(self, request_id: str, payload: str) -> None:
        await self.client.set(
            f"opcenter:chat:request:{request_id}", payload, ex=self.request_ttl
        )

    async def pop_request(self, request_id: str) -> str | None:
        return await self.client.getdel(f"opcenter:chat:request:{request_id}")
