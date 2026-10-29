"""Runs one checkpointed agent task per user message."""

import asyncio
import logging
import time

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, ToolCallPart
from pydantic_ai.usage import UsageLimits

from .agent import Deps
from .context import ContextProcessor, TaskEvidence
from .store import DONE, FAILED, EventStore


logger = logging.getLogger(__name__)

STEP_LIMIT = "step_limit"
WAKE_PROMPT = (
    "[NO NEW USER INPUT. CONTINUE AUTONOMOUS WORK. ONLY USE send FOR GENUINELY "
    "NEW INFORMATION OR WHEN USER INPUT IS NEEDED.]"
)


class Runner:
    def __init__(
        self,
        agent,
        store: EventStore,
        processor: ContextProcessor,
        task_evidence: TaskEvidence,
        *,
        max_steps: int = 50,
        max_feedback_chars: int = 50000,
        append_history=None,
    ):
        self._agent = agent
        self._store = store
        self._processor = processor
        self._evidence = task_evidence
        self._max_steps = int(max_steps)
        self._max_feedback = int(max_feedback_chars)
        self._append_history = append_history

    def run_task(self, message: str) -> str:
        task_id = self._store.start_task(message)
        self._evidence.reset()
        return asyncio.run(self._run(task_id, message, message, None))

    def resume_interrupted(self) -> str | None:
        """Continues the newest task left running by a crash."""
        task = self._store.interrupted_task()
        if task is None:
            return None
        history = resumable_history(task.history)
        self._evidence.reset()
        for event in self._store.events(task.id, "tool_result"):
            self._evidence.add(event["call_id"], event["name"], event["arguments"], event["result"])
        self._store.record(task.id, "resumed", messages=len(history))
        logger.info("[RESUME] task=%s messages=%s", task.id, len(history))
        prompt = None if history else task.message
        return asyncio.run(self._run(task.id, task.message, prompt, history or None))

    async def _run(self, task_id, message, prompt, history) -> str:
        self._processor.task_message = message
        deps = Deps(
            task_id=task_id,
            store=self._store,
            evidence=self._evidence,
            processor=self._processor,
            max_steps=self._max_steps,
            max_feedback_chars=self._max_feedback,
        )
        output = ""
        status = DONE
        try:
            async with self._agent.iter(
                prompt,
                message_history=history,
                deps=deps,
                usage_limits=UsageLimits(request_limit=self._max_steps),
            ) as run:
                async for _node in run:
                    self._store.save_history(
                        task_id, ModelMessagesTypeAdapter.dump_json(run.all_messages())
                    )
                output = str(run.result.output) if run.result is not None else ""
                self._store.save_history(
                    task_id, ModelMessagesTypeAdapter.dump_json(run.all_messages())
                )
        except UsageLimitExceeded as error:
            self._store.record(task_id, STEP_LIMIT, error=str(error))
            logger.warning("[STEP_LIMIT] task=%s %s", task_id, error)
        except Exception as error:
            status = FAILED
            self._store.record(task_id, "error", error=f"{type(error).__name__}: {error}")
            logger.exception("[TASK_FAILED] task=%s", task_id)
        self._store.record(task_id, "finished", status=status, output=output)
        self._store.finish_task(task_id, status)
        if self._append_history is not None:
            self._append_history(message, self._store.events(task_id, "tool_result"), output)
        return output

    def serve(self, receive, *, should_stop=lambda: False, poll_seconds=0.2,
              wakeup_seconds=600.0, clock=time.monotonic, sleep=time.sleep):
        """Runs a task for each incoming message until should_stop() is true."""
        self.resume_interrupted()
        last_activity = clock()
        while not should_stop():
            message = str(receive() or "").strip()
            if message:
                self.run_task(message)
                last_activity = clock()
            elif clock() - last_activity >= wakeup_seconds:
                self.run_task(WAKE_PROMPT)
                last_activity = clock()
            else:
                sleep(poll_seconds)


def resumable_history(history: bytes | None) -> list:
    """Returns saved messages without a trailing unanswered tool call."""
    if not history:
        return []
    messages = list(ModelMessagesTypeAdapter.validate_json(history))
    if messages and isinstance(messages[-1], ModelResponse) and any(
        isinstance(part, ToolCallPart) for part in messages[-1].parts
    ):
        messages.pop()
    return messages
