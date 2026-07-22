import asyncio

from src.request_store import RedisRequestStore


class FakeRedis:
    def __init__(self):
        self.values = {}

    async def eval(self, _script, _keys, key, owner, _ttl):
        current = self.values.get(key)
        if current not in {None, owner}:
            return 0
        self.values[key] = owner
        return 1

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def getdel(self, key):
        return self.values.pop(key, None)


def test_request_store_is_single_use_and_enforces_thread_owner() -> None:
    async def exercise():
        store = RedisRequestStore(FakeRedis(), request_ttl=30, ownership_ttl=60)
        assert await store.claim_thread("thread", "session-a")
        assert await store.claim_thread("thread", "session-a")
        assert not await store.claim_thread("thread", "session-b")
        await store.save_request("request", '{"message":"hello"}')
        assert await store.pop_request("request") == '{"message":"hello"}'
        assert await store.pop_request("request") is None

    asyncio.run(exercise())
