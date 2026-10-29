"""Tests the Python runtime end to end with a scripted model."""

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

import evidence
from harness.agent import build_agent
from harness.context import ContextProcessor, PromptSections, TaskEvidence
from harness.metta import ToolResult
from harness.runner import Runner
from harness.skills import SkillSpec
from harness.store import DONE, RUNNING, EventStore


SKILLS = (
    SkillSpec("send", "Send message to user", ("string",)),
    SkillSpec("websearch", "Search the web", ("string",)),
)


def sections():
    return PromptSections(
        prompt="You are OmegaClaw.",
        rules="Use send to talk to the user.",
        prompt_extensions="",
        memory_directory="/repo/memory",
        history='("2026-10-28 09:00:00" "HUMAN_MESSAGE: earlier question")',
        current_time="2026-10-28 10:00:00",
    )


class Skills:
    """Fakes MeTTa skills and records their calls."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, name, args):
        self.calls.append((name, args))
        if name == "websearch":
            return ToolResult(True, self.results.get(args[0], f"results for {args[0]}"))
        if name == "send":
            return ToolResult(True, "SEND-SUCCESS")
        return ToolResult(False, "", f"unknown skill {name}")


class Script:
    """Plays scripted model responses and records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []

    def __call__(self, messages, info):
        self.seen.append(list(messages))
        return self.responses.pop(0)


def call(tool, call_id, **args):
    return ModelResponse(parts=[ToolCallPart(tool_name=tool, args=args, tool_call_id=call_id)])


def answer(text):
    return ModelResponse(parts=[TextPart(content=text)])


def make_runner(tmp_path, script, skills, *, window=4000, max_steps=10, store=None):
    task_evidence = TaskEvidence()
    processor = ContextProcessor(
        sections, task_evidence, context_window_tokens=window, max_output_tokens=200
    )
    agent = build_agent(
        FunctionModel(script), SKILLS, skills, processor, max_output_tokens=200
    )
    store = store or EventStore(str(tmp_path / "harness.sqlite"))
    runner = Runner(agent, store, processor, task_evidence, max_steps=max_steps)
    return runner, store, processor


def text_of(message):
    return "\n".join(str(getattr(part, "content", "")) for part in message.parts)


@pytest.fixture(autouse=True)
def clean_evidence():
    evidence.reset()
    yield
    evidence.reset()


def test_task_runs_skills_and_records_events(tmp_path):
    script = Script(call("websearch", "c1", string="omega"), call("send", "c2", string="found it"), answer("done"))
    skills = Skills()
    runner, store, _ = make_runner(tmp_path, script, skills)

    output = runner.run_task("look up omega")

    assert output == "done"
    assert skills.calls == [("websearch", ["omega"]), ("send", ["found it"])]
    task = store.task(1)
    assert task.status == DONE and task.history
    events = store.events(1, "tool_result")
    assert [(e["call_id"], e["name"], e["result"]) for e in events] == [
        ("c1", "websearch", "results for omega"),
        ("c2", "send", "SEND-SUCCESS"),
    ]


def test_first_request_is_compiled_from_prompt_history_and_task(tmp_path):
    script = Script(answer("ok"))
    runner, _, _ = make_runner(tmp_path, script, Skills())

    runner.run_task("write the summary")

    (request,) = script.seen[0]
    assert isinstance(request, ModelRequest)
    system = [p for p in request.parts if isinstance(p, SystemPromptPart)]
    user = [p for p in request.parts if isinstance(p, UserPromptPart)]
    assert len(system) == 1 and "You are OmegaClaw." in system[0].content
    assert "Use send to talk to the user." in system[0].content
    assert len(user) == 1
    assert "earlier question" in user[0].content
    assert user[0].content.rstrip().endswith("write the summary")


def test_latest_step_stays_native_and_older_results_become_evidence(tmp_path):
    script = Script(
        call("websearch", "c1", string="alpha"),
        call("websearch", "c2", string="beta"),
        answer("done"),
    )
    runner, _, _ = make_runner(tmp_path, script, Skills())

    runner.run_task("compare alpha and beta")

    second, third = script.seen[1], script.seen[2]
    assert isinstance(second[1], ModelResponse) and second[1].parts[0].tool_call_id == "c1"
    assert isinstance(second[2].parts[0], ToolReturnPart)
    assert sum("results for alpha" in text_of(m) for m in second) == 1
    assert "results for alpha" in text_of(third[0])
    assert "websearch" in text_of(third[0])
    assert third[1].parts[0].tool_call_id == "c2"
    assert "results for beta" not in text_of(third[0])
    assert "CURRENT_TASK:\ncompare alpha and beta" in text_of(third[0])


def test_tool_failures_are_returned_to_the_model(tmp_path):
    def failing(name, args):
        raise RuntimeError("bridge down")

    script = Script(call("websearch", "c1", string="x"), answer("gave up"))
    runner, store, _ = make_runner(tmp_path, script, failing)

    runner.run_task("search")

    returned = script.seen[1][-1].parts[0]
    assert returned.content == "ERROR: RuntimeError: bridge down"
    assert store.events(1, "tool_result")[0]["ok"] is False


def test_step_limit_stops_the_task(tmp_path):
    script = Script(*[call("websearch", f"c{i}", string=str(i)) for i in range(10)])
    runner, store, _ = make_runner(tmp_path, script, Skills(), max_steps=3)

    runner.run_task("loop forever")

    assert store.task(1).status == DONE
    assert len(store.events(1, "step_limit")) == 1
    assert len(script.seen) == 3


def test_omitted_evidence_can_be_recalled(tmp_path):
    big = {"alpha": "alpha " * 300, "beta": "beta " * 300}
    script = Script(
        call("websearch", "c1", string="alpha"),
        call("websearch", "c2", string="beta"),
        call("send", "c3", string="progress"),
        call("recall", "c4", ids="tool-result-1"),
        answer("done"),
    )
    runner, store, processor = make_runner(tmp_path, script, Skills(big), window=1300)

    runner.run_task("compare")

    head = text_of(script.seen[3][0])
    assert "beta beta" in head
    assert "TOOL_RESULT_OMITTED id=tool-result-1" in head
    recall_result = store.events(1, "tool_result")[3]["result"]
    assert recall_result.startswith("RECALL-QUEUED")
    head = text_of(script.seen[4][0])
    assert "alpha alpha" in head


class Crash(BaseException):
    """Simulates the process dying mid-task."""


def test_interrupted_task_resumes_from_its_last_step(tmp_path):
    def crashing(name, args):
        if name == "send":
            raise Crash()
        return Skills()(name, args)

    first_script = Script(call("websearch", "c1", string="alpha"), call("send", "c2", string="partial"))
    runner, store, _ = make_runner(tmp_path, first_script, crashing)
    with pytest.raises(Crash):
        runner.run_task("research alpha")
    assert store.interrupted_task().status == RUNNING

    evidence.reset()
    second_script = Script(call("send", "c3", string="resumed"), answer("done"))
    resumed, _, _ = make_runner(tmp_path, second_script, Skills(), store=store)

    assert resumed.resume_interrupted() == "done"
    assert store.task(1).status == DONE
    resumed_request = second_script.seen[0]
    assert "results for alpha" in "\n".join(text_of(m) for m in resumed_request)
    assert [e["kind"] for e in store.events(1) if e["kind"] == "resumed"] == ["resumed"]


def test_oversized_latest_step_falls_back_to_prompt_and_task(tmp_path):
    script = Script(call("websearch", "c1", string="huge"), answer("done"))
    skills = Skills({"huge": "word " * 3000})
    runner, store, processor = make_runner(tmp_path, script, skills, window=1300)

    assert runner.run_task("read the huge page") == "done"

    head = script.seen[1][0]
    assert "You are OmegaClaw." in text_of(head)
    assert "CURRENT_TASK:\nread the huge page" in text_of(head)
    assert "earlier question" not in text_of(head)
    assert processor.last_manifest is None
    assert store.task(1).status == DONE


def test_no_interrupted_task_means_nothing_to_resume(tmp_path):
    runner, _, _ = make_runner(tmp_path, Script(), Skills())

    assert runner.resume_interrupted() is None
