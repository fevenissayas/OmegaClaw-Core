"""Mindplex context plugin.

Every LLM request built by the Omega loop passes through process_request()
(registered via providers.registerRequestProcessor). It keeps tool results as
task evidence across turns (task_evidence) and compiles the system prompt,
older evidence and whole history records into the model's token budget
(context_compiler). The latest step's native tool-call messages and the
loop's turn message are passed through unchanged, after the compiled
context, with their tokens reserved in the budget. The recall skill
brings evidence the budget left out back on the next turn.

Any failure inside the processor leaves the request unchanged (the hook in
providers.py logs it), so the plugin can never stop the agent loop.
"""

import json
import logging
import time

import providers
from config import config_get_by_key
from providers import (
    LLMMessage,
    LLMRequest,
    LLMToolCallMessage,
    LLMToolCallResponseMessage,
)

import task_evidence
from context_compiler import (
    ContextBudgetError,
    ContextCompiler,
    ContextInput,
    estimate_tokens,
    loop_context_records,
    validate_context_budget,
)


logger = logging.getLogger(__name__)

PROCESSOR_ID = "mindplex-context"
NO_NEW_INPUT_PREFIX = "[NO NEW USER INPUT"
SUCCESS_PREFIX = "SUCCESS, RETURN: "
CALL_HEADER_MAX_CHARS = 300
# Markers written by getContext in src/loop.metta, in order.
_PROMPT = "PROMPT: "
_SKILLS = " SKILLS: "
_MEMORY_DIRECTORY = " SAVE_PERMANENT_FILES_DIR: "
_HISTORY = " HISTORY: "
_TIME = " TIME: "
_OPTIONAL_EMPTY_KINDS = {"PROMPT_EXTENSIONS", "OUTPUT_FORMAT"}

_context_window_tokens = 32768
_max_new_input_loops: int | None = None
_wakeup_interval_seconds = 600.0
_max_feedback_chars = 50000
_clock = time.monotonic

_task_message = ""
_task_interaction = 0
_last_request_at: float | None = None
_last_manifest: dict[str, object] | None = None
_last_task_telemetry: dict[str, object] | None = None


def start():
    """Read configuration, validate the budget and register the processor."""
    configure(
        context_window_tokens=config_get_by_key("contextWindowTokens", 32768),
        max_output_tokens=config_get_by_key("maxOutputToken", 6000),
        max_new_input_loops=config_get_by_key("maxNewInputLoops", 50),
        wakeup_interval_seconds=config_get_by_key("wakeupInterval", 600),
        max_feedback_chars=config_get_by_key("maxFeedback", 50000),
    )
    providers.registerRequestProcessor(PROCESSOR_ID, process_request)
    return True


def configure(
    *,
    context_window_tokens,
    max_output_tokens,
    max_new_input_loops=None,
    wakeup_interval_seconds=600,
    max_feedback_chars=50000,
):
    """Set the plugin's budget and loop limits; also used by tests."""
    global _context_window_tokens, _max_new_input_loops
    global _wakeup_interval_seconds, _max_feedback_chars
    window = int(context_window_tokens)
    output = int(max_output_tokens)
    budget = validate_context_budget(window, output)
    _context_window_tokens = window
    _max_new_input_loops = (
        int(max_new_input_loops) if max_new_input_loops is not None else None
    )
    _wakeup_interval_seconds = float(wakeup_interval_seconds)
    _max_feedback_chars = int(max_feedback_chars)
    allocation = {
        "context_window_tokens": window,
        "max_output_tokens": output,
        "input_token_budget": budget,
        "max_new_input_loops": _max_new_input_loops,
    }
    logger.info("[CONTEXT_BUDGET] %s", json.dumps(allocation, separators=(",", ":")))
    if budget < output:
        logger.warning(
            "[CONTEXT_BUDGET_WARNING] input reserve is smaller than output reserve: %s",
            json.dumps(allocation, separators=(",", ":")),
        )
    return budget


def reset():
    """Forget the active task and all task evidence."""
    global _task_message, _task_interaction, _last_request_at
    global _last_manifest, _last_task_telemetry
    task_evidence.reset()
    _task_message = ""
    _task_interaction = 0
    _last_request_at = None
    _last_manifest = None
    _last_task_telemetry = None


def process_request(request: LLMRequest) -> LLMRequest:
    """Compile one loop request; see the module docstring."""
    global _task_interaction, _last_request_at, _last_manifest, _last_task_telemetry
    if not request.messages:
        raise ValueError("request has no messages")
    sections = parse_system_context(decode(str(request.messages[0].content)))
    episodes = list(request.messages[1:])
    turn = episodes.pop() if episodes and _is_user_message(episodes[-1]) else None
    turn_text = decode(str(turn.content)) if turn is not None else ""

    now = _clock()
    idle = _last_request_at is not None and now - _last_request_at >= _wakeup_interval_seconds
    _last_request_at = now
    if turn_text and not turn_text.startswith(NO_NEW_INPUT_PREFIX):
        _start_task(turn_text)
        current_ids: set[str] = set()
    elif idle:
        # The loop wakes itself after wakeupInterval: a new task generation.
        _start_task("")
        current_ids = set()
    else:
        current_ids = _harvest(episodes)
    _task_interaction += 1

    count_tokens = getattr(providers._llmprovider, "count_tokens", None) or estimate_tokens
    evidence_records = tuple(
        record for record in task_evidence.records() if record.id not in current_ids
    )
    evidence_sources = tuple(
        source
        for source in task_evidence.sources()
        if not set(source.chunk_ids) <= current_ids
    )
    records = tuple(
        record
        for record in loop_context_records(
            prompt=sections["prompt"],
            skills=sections["skills"],
            prompt_extensions="",
            output_format="",
            memory_directory=sections["memory_directory"],
            current_time=sections["time"],
            evidence_records=evidence_records,
            history=sections["history"],
            evidence_sources=evidence_sources,
        )
        if record.text.strip() or record.kind not in _OPTIONAL_EMPTY_KINDS
    )
    native_tokens = sum(_message_tokens(message, count_tokens) for message in episodes)
    evidence_stats = task_evidence.stats()
    try:
        compiled = ContextCompiler(count_tokens).compile(
            ContextInput(
                records=records,
                task_message=_task_message,
                turn_message=turn_text,
                context_window_tokens=_context_window_tokens - native_tokens,
                max_output_tokens=int(request.max_tokens),
                reasoning_mode=str(request.reasoning_mode),
                separate_turn=True,
            )
        )
    except ContextBudgetError:
        logger.exception(
            "[CONTEXT_BUDGET_FAILURE] task_generation=%s interaction=%s "
            "context_window_tokens=%s native_tokens=%s max_output_tokens=%s evidence=%s",
            evidence_stats.task_generation,
            _task_interaction,
            _context_window_tokens,
            native_tokens,
            request.max_tokens,
            json.dumps(evidence_stats.as_dict(), separators=(",", ":")),
        )
        raise
    task_evidence.clear_recall_preferences()

    _last_manifest = compiled.manifest.as_dict()
    _last_manifest["native_message_tokens"] = native_tokens
    _last_task_telemetry = {
        "task_generation": evidence_stats.task_generation,
        "interaction": _task_interaction,
        "evidence": evidence_stats.as_dict(),
        "context": _last_manifest,
    }
    logger.info("[CONTEXT_MANIFEST] %s", json.dumps(_last_manifest, separators=(",", ":")))
    logger.info("[TASK_TELEMETRY] %s", json.dumps(_last_task_telemetry, separators=(",", ":")))
    manifest = compiled.manifest
    if manifest.estimated_input_tokens * 10 >= manifest.input_token_budget * 9:
        logger.warning(
            "[CONTEXT_BUDGET_PRESSURE] task_generation=%s interaction=%s "
            "estimated_input_tokens=%s input_token_budget=%s omitted_records=%s",
            evidence_stats.task_generation,
            _task_interaction,
            manifest.estimated_input_tokens,
            manifest.input_token_budget,
            len(manifest.omitted_record_ids),
        )

    # system, the context records and current task, the latest native tool
    # step, then the loop's own turn message unchanged.
    compiled_messages = compiled.request.messages
    if turn_text.strip():
        compiled_messages = compiled_messages[:-1]
    messages = [
        LLMMessage().with_role(message.role).with_content(message.content)
        for message in compiled_messages
    ]
    messages += episodes
    if turn is not None:
        messages.append(turn)
    return (
        LLMRequest()
        .with_messages(messages)
        .with_max_tokens(request.max_tokens)
        .with_reasoning_mode(request.reasoning_mode)
        .with_tools(list(request.tools))
    )


def recall(record_ids):
    """Queue exact evidence ids for the next request, within the active budget."""
    evidence_stats = task_evidence.stats()
    telemetry_is_current = bool(
        _last_task_telemetry
        and _last_task_telemetry.get("task_generation") == evidence_stats.task_generation
    )
    manifest = _last_manifest if telemetry_is_current else None
    input_budget = (
        int(manifest["input_token_budget"])
        if manifest and "input_token_budget" in manifest
        else None
    )
    interactions_remaining = (
        max(0, _max_new_input_loops - _task_interaction)
        if _max_new_input_loops is not None and telemetry_is_current
        else None
    )
    counter = getattr(providers._llmprovider, "count_tokens", None) or estimate_tokens
    return task_evidence.recall(
        record_ids,
        input_token_budget=input_budget,
        count_tokens=counter,
        interactions_remaining=interactions_remaining,
    )


def lastContextManifest():
    return json.dumps(_last_manifest or {}, separators=(",", ":"))


def lastTaskTelemetry():
    return json.dumps(_last_task_telemetry or {}, separators=(",", ":"))


def parse_system_context(text: str) -> dict[str, str]:
    """Split the getContext() system prompt into its sections."""
    if not text.startswith(_PROMPT):
        raise ValueError("system prompt does not start with PROMPT:")
    skills = text.find(_SKILLS, len(_PROMPT))
    memory_directory = text.find(_MEMORY_DIRECTORY, max(skills, 0))
    history = text.find(_HISTORY, max(memory_directory, 0))
    current_time = text.rfind(_TIME)
    if min(skills, memory_directory, history) < 0 or current_time < history:
        raise ValueError("system prompt does not have the getContext() sections")
    return {
        "prompt": text[len(_PROMPT):skills].strip(),
        "skills": text[skills + len(_SKILLS):memory_directory].strip(),
        "memory_directory": text[memory_directory + len(_MEMORY_DIRECTORY):history].strip(),
        "history": text[history + len(_HISTORY):current_time].strip(),
        "time": text[current_time + len(_TIME):].strip(),
    }


def decode(text: str) -> str:
    """Undo string-safe from src/utils.metta."""
    return (
        text.replace("_quote_", '"')
        .replace("_newline_", "\n")
        .replace("_apostrophe_", "'")
    )


def _start_task(message: str) -> None:
    global _task_message, _task_interaction
    task_evidence.reset()
    _task_message = message
    _task_interaction = 0


def _harvest(episodes) -> set[str]:
    """Store the previous step's tool results; return their evidence ids."""
    calls = {
        call.id: call
        for message in episodes
        if isinstance(message, LLMToolCallMessage)
        for call in (message.calls or [])
    }
    before = {record.id for record in task_evidence.records()}
    for message in episodes:
        if not isinstance(message, LLMToolCallResponseMessage):
            continue
        content = decode(str(message.content or ""))
        if content.startswith(SUCCESS_PREFIX):
            content = content[len(SUCCESS_PREFIX):]
        task_evidence.append(
            content,
            _max_feedback_chars,
            header=_call_header(message.callid, calls.get(message.callid)),
        )
    return {record.id for record in task_evidence.records()} - before


def _call_header(callid, call) -> str:
    if call is None:
        return f"[TOOL_CALL id={callid}]"
    arguments = json.dumps(call.arguments, ensure_ascii=False)
    header = f"[TOOL_CALL id={callid}] {call.name} {arguments}"
    if len(header) > CALL_HEADER_MAX_CHARS:
        header = header[: CALL_HEADER_MAX_CHARS - 1] + "…"
    return header


def _is_user_message(message) -> bool:
    return type(message) is LLMMessage and message.role == "user"


def _message_tokens(message, count_tokens) -> int:
    text = str(message.content or "")
    if isinstance(message, LLMToolCallMessage):
        text += " " + " ".join(
            f"{call.name} {json.dumps(call.arguments, ensure_ascii=False)}"
            for call in (message.calls or [])
        )
    return 4 + max(0, int(count_tokens(text)))
