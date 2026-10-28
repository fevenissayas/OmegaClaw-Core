"""Dave's context compiler as a PydanticAI history processor.

Before every model request the processor rebuilds the history as

    [compiled request: system prompt, history records, older evidence, task]
    + [latest tool-call response and its tool returns, unchanged]

Older tool results live in the task evidence store (src/evidence.py), so
the compiler can rank them against history within the token budget and
leave omission markers the model can recall by id. The latest step stays
native so tool calls and their returns remain paired.
"""

from dataclasses import dataclass
import json
import logging
from typing import Callable

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

import evidence
from context import (
    ContextBudgetError,
    ContextCompiler,
    ContextInput,
    estimate_tokens,
    loop_context_records,
)


logger = logging.getLogger(__name__)

_OPTIONAL_EMPTY_KINDS = {"PROMPT_EXTENSIONS", "OUTPUT_FORMAT"}


@dataclass(frozen=True)
class PromptSections:
    """The non-evidence parts of the context, read from MeTTa each step."""

    prompt: str
    rules: str
    prompt_extensions: str
    memory_directory: str
    history: str
    current_time: str


class TaskEvidence:
    """Tool results of the active task, keyed by tool call id."""

    def __init__(self, max_chars: int = 50000):
        self._max_chars = int(max_chars)
        self._ids_by_call: dict[str, set[str]] = {}

    def reset(self) -> None:
        evidence.reset()
        self._ids_by_call.clear()

    def add(self, call_id: str, tool_name: str, args: dict, result: str) -> set[str]:
        before = {record.id for record in evidence.records()}
        evidence.append(command_return(tool_name, args, result), self._max_chars)
        ids = {record.id for record in evidence.records()} - before
        self._ids_by_call.setdefault(call_id, set()).update(ids)
        return ids

    def ids_for_calls(self, call_ids) -> set[str]:
        return set().union(*(self._ids_by_call.get(call_id, set()) for call_id in call_ids))


class ContextProcessor:
    """history_processors entry for the runtime's Agent."""

    def __init__(
        self,
        sections: Callable[[], PromptSections],
        task_evidence: TaskEvidence,
        *,
        context_window_tokens: int,
        max_output_tokens: int,
        count_tokens: Callable[[str], int] = estimate_tokens,
    ):
        self._sections = sections
        self._evidence = task_evidence
        self._window = int(context_window_tokens)
        self._max_output = int(max_output_tokens)
        self._count = count_tokens
        self.task_message = ""
        self.last_manifest: dict | None = None

    def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        tail = latest_tool_step(messages)
        tail_call_ids = {
            part.tool_call_id
            for message in tail
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        hidden = self._evidence.ids_for_calls(tail_call_ids)
        sections = self._sections()
        records = tuple(
            record
            for record in loop_context_records(
                prompt=sections.prompt,
                skills=sections.rules,
                prompt_extensions=sections.prompt_extensions,
                output_format="",
                memory_directory=sections.memory_directory,
                current_time=sections.current_time,
                evidence_records=tuple(r for r in evidence.records() if r.id not in hidden),
                history=sections.history,
                evidence_sources=tuple(
                    s for s in evidence.sources() if not set(s.chunk_ids) <= hidden
                ),
            )
            if record.text.strip() or record.kind not in _OPTIONAL_EMPTY_KINDS
        )
        tail_tokens = sum(message_tokens(message, self._count) for message in tail)
        # The first step's turn is the task itself; later steps continue it.
        turn = "" if tail else self.task_message
        try:
            compiled = ContextCompiler(self._count).compile(
                ContextInput(
                    records=records,
                    task_message=self.task_message,
                    turn_message=turn,
                    context_window_tokens=self._window - tail_tokens,
                    max_output_tokens=self._max_output,
                )
            )
        except ContextBudgetError:
            # The latest step alone leaves no room for the required records:
            # send prompt and task without history or evidence, and let the
            # provider decide whether the request still fits.
            logger.exception("[CONTEXT_BUDGET_FAILURE] native_message_tokens=%s", tail_tokens)
            self.last_manifest = None
            head = ModelRequest(
                parts=[
                    SystemPromptPart(content=f"{sections.prompt}\n\n{sections.rules}".strip()),
                    UserPromptPart(content=f"CURRENT_TASK:\n{self.task_message}"),
                ]
            )
            return [head, *tail]
        evidence.clear_recall_preferences()
        self.last_manifest = compiled.manifest.as_dict()
        self.last_manifest["native_message_tokens"] = tail_tokens
        logger.info("[CONTEXT_MANIFEST] %s", json.dumps(self.last_manifest, separators=(",", ":")))

        system, user = compiled.request.messages
        head = ModelRequest(
            parts=[
                SystemPromptPart(content=system.text_content()),
                UserPromptPart(content=user.text_content()),
            ]
        )
        return [head, *tail]

    def recall(self, record_ids: str, interactions_remaining: int | None = None) -> str:
        budget = self.last_manifest.get("input_token_budget") if self.last_manifest else None
        return evidence.recall(
            record_ids,
            input_token_budget=budget,
            count_tokens=self._count,
            interactions_remaining=interactions_remaining,
        )


def latest_tool_step(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The last response with tool calls and everything after it."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ModelResponse) and any(
            isinstance(part, ToolCallPart) for part in message.parts
        ):
            return list(messages[index:])
    return []


def message_tokens(message: ModelMessage, count_tokens: Callable[[str], int]) -> int:
    texts = []
    for part in message.parts:
        if isinstance(part, ToolCallPart):
            texts.append(f"{part.tool_name} {part.args_as_json_str()}")
        elif isinstance(part, ToolReturnPart):
            texts.append(part.model_response_str())
        else:
            texts.append(str(getattr(part, "content", "")))
    return 4 + sum(max(0, int(count_tokens(text))) for text in texts)


def command_return(tool_name: str, args: dict, result: str) -> str:
    """Serialize a tool result the way the MeTTa loop logged it, so the
    evidence store's source-page detection and labels keep working."""
    call = " ".join([tool_name, *(_quoted(value) for value in args.values())])
    return f"(COMMAND_RETURN: (({call}) {_quoted(result)}))"


def _quoted(value) -> str:
    return '"' + str(value).replace('"', '\\"') + '"'
