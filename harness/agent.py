"""PydanticAI agent wired to MeTTa skills, task evidence and the event log."""

from dataclasses import dataclass
import logging

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.settings import ModelSettings

from .context import ContextProcessor, TaskEvidence
from .metta import ToolResult
from .skills import SkillSpec
from .store import EventStore


logger = logging.getLogger(__name__)


@dataclass
class Deps:
    task_id: int
    store: EventStore
    evidence: TaskEvidence
    processor: ContextProcessor
    max_steps: int
    max_feedback_chars: int = 50000


def build_agent(
    model,
    skills: tuple[SkillSpec, ...],
    execute_skill,
    processor: ContextProcessor,
    *,
    max_output_tokens: int,
) -> Agent:
    """Agent with one typed tool per MeTTa skill plus recall.

    execute_skill(name, args) runs a skill and returns a ToolResult; it is
    the MettaBridge in production and a fake in tests.
    """
    tools = [skill_tool(spec, execute_skill) for spec in skills if spec.name != "recall"]
    tools.append(Tool(recall, takes_ctx=True, name="recall"))
    return Agent(
        model,
        deps_type=Deps,
        output_type=str,
        tools=tools,
        # Runs before every model request; the processed history also
        # replaces the run's stored history (what the runner checkpoints).
        capabilities=[ProcessHistory(processor)],
        model_settings=ModelSettings(max_tokens=int(max_output_tokens)),
    )


def skill_tool(spec: SkillSpec, execute_skill) -> Tool:
    schema = {
        "type": "object",
        "properties": {param: {"type": "string"} for param in spec.params},
        "required": list(spec.params),
        "additionalProperties": False,
    }

    def run_skill(ctx: RunContext[Deps], **arguments) -> str:
        args = [str(arguments.get(param, "")) for param in spec.params]
        try:
            result = execute_skill(spec.name, args)
        except Exception as error:
            result = ToolResult(False, "", f"{type(error).__name__}: {error}")
        return record_result(ctx, spec.name, arguments, result)

    return Tool.from_schema(
        function=run_skill,
        name=spec.tool_name,
        description=spec.description,
        json_schema=schema,
        takes_ctx=True,
    )


def recall(ctx: RunContext[Deps], ids: str) -> str:
    """Bring task evidence that was left out of context back on the next step,
    by the exact ids shown in TOOL_RESULT_OMITTED or SOURCE markers, comma separated."""
    remaining = max(0, ctx.deps.max_steps - ctx.usage.requests)
    result = ToolResult(True, ctx.deps.processor.recall(ids, remaining))
    return record_result(ctx, "recall", {"ids": ids}, result)


def record_result(ctx: RunContext[Deps], name: str, arguments: dict, result: ToolResult) -> str:
    text = result.for_model()
    limit = ctx.deps.max_feedback_chars
    if len(text) > limit:
        text = text[-limit:]
    call_id = ctx.tool_call_id or ""
    ctx.deps.evidence.add(call_id, name, arguments, text)
    ctx.deps.store.record(
        ctx.deps.task_id,
        "tool_result",
        call_id=call_id,
        name=name,
        arguments=arguments,
        ok=result.ok,
        result=text,
    )
    logger.info("[TOOL] %s ok=%s chars=%s", name, result.ok, len(text))
    return text
