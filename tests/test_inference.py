import asyncio

import pytest

from src.inference import InferenceBusyError, InferenceGate


def test_inference_gate_rejects_an_unbounded_waiting_queue() -> None:
    async def exercise():
        gate = InferenceGate("test")
        gate.semaphore = asyncio.Semaphore(1)
        gate.max_queue_depth = 1
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def first():
            async with gate.slot():
                first_entered.set()
                await release_first.wait()

        first_task = asyncio.create_task(first())
        await first_entered.wait()
        second_task = asyncio.create_task(gate.semaphore.acquire())
        await asyncio.sleep(0)
        gate.waiters = 1
        with pytest.raises(InferenceBusyError):
            async with gate.slot():
                pass
        second_task.cancel()
        release_first.set()
        await first_task

    asyncio.run(exercise())
