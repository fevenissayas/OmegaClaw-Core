"""Mindplex context plugin: request processor hook and loop request compilation."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "plugins" / "mindplex"))

import pytest

import providers
from providers import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMTool,
    LLMToolCall,
    LLMToolCallMessage,
    LLMToolCallResponseMessage,
    LLMToolParameter,
    llmRequest,
    llmRequestMessage,
    llmToolCallMessage,
    llmToolCallResponseMessage,
)

import mindplex
import task_evidence

NO_INPUT = "[NO NEW USER INPUT. CONTINUE AUTONOMOUS WORK.]"
OUTPUT_TOKENS = 50


def words(text):
    return len(text.split())


class FakeProvider:
    def __init__(self):
        self.requests = []

    def chat(self, request):
        self.requests.append(request)
        call = LLMToolCall().with_name("send").with_id("call-1").with_arguments({"content": "hi"})
        return LLMResponse().add_tool_call(call)

    def count_tokens(self, text):
        return words(text)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(providers, "_llmprovider", fake)
    monkeypatch.setattr(providers, "_requestProcessors", {})
    return fake


@pytest.fixture
def clock(monkeypatch, provider):
    tick = Clock()
    monkeypatch.setattr(mindplex, "_clock", tick)
    mindplex.reset()
    mindplex.configure(
        context_window_tokens=2000,
        max_output_tokens=OUTPUT_TOKENS,
        max_new_input_loops=10,
        wakeup_interval_seconds=600,
        max_feedback_chars=50000,
    )
    yield tick
    mindplex.reset()


def send_tool():
    return LLMTool().with_name("send").with_description("send").add_parameter(
        LLMToolParameter().with_name("content")
    )


def system_prompt(history="", time="2026-10-28 10:00:00"):
    return (
        "PROMPT: You are Omega. SKILLS: (send content)_newline_ext rule"
        f" SAVE_PERMANENT_FILES_DIR: /repo/memory HISTORY: {history} TIME: {time}"
    )


def history_record(stamp, text):
    return f'("{stamp}" "HUMAN_MESSAGE: {text}")\n'


def loop_request(*episodes, history=""):
    return llmRequest(
        llmRequestMessage("system", system_prompt(history)),
        list(episodes),
        OUTPUT_TOKENS,
        "medium",
        [send_tool()],
    )


def user(text):
    return llmRequestMessage("user", text)


def tool_step(callid, name, result, **arguments):
    call = LLMToolCall().with_name(name).with_id(callid).with_arguments(arguments)
    return [
        llmToolCallMessage("assistant", [call], "[TOOL CALL]"),
        llmToolCallResponseMessage("tool", callid, f"SUCCESS, RETURN: {result}"),
    ]


def contents(request):
    return [message.content for message in request.messages]


# providers.registerRequestProcessor


def test_processor_replaces_request(provider):
    replacement = LLMRequest().add_message(llmRequestMessage("system", "compiled"))
    providers.registerRequestProcessor("test", lambda request: replacement)

    providers.llmProviderChat(loop_request(user("hi")))

    assert provider.requests == [replacement]


def test_failing_processor_leaves_request_unchanged(provider):
    def broken(request):
        raise RuntimeError("boom")

    providers.registerRequestProcessor("broken", broken)
    original = loop_request(user("hi"))

    response = providers.llmProviderChat(original)

    assert provider.requests == [original]
    assert not response.is_error()


def test_processor_returning_wrong_type_is_ignored(provider):
    providers.registerRequestProcessor("wrong", lambda request: "not a request")
    original = loop_request(user("hi"))

    providers.llmProviderChat(original)

    assert provider.requests == [original]


def test_processors_run_in_registration_order(provider):
    seen = []

    def first(request):
        seen.append("first")
        return request.add_message(llmRequestMessage("user", "a"))

    def second(request):
        seen.append("second")
        return request.add_message(llmRequestMessage("user", "b"))

    providers.registerRequestProcessor("first", first)
    providers.registerRequestProcessor("second", second)
    providers.llmProviderChat(loop_request(user("hi")))

    assert seen == ["first", "second"]
    assert contents(provider.requests[0])[-2:] == ["a", "b"]


# mindplex.parse_system_context


def test_parse_system_context_splits_get_context_sections():
    sections = mindplex.parse_system_context(
        mindplex.decode(system_prompt(history_record("2026-10-28 09:00:00", "hello")))
    )

    assert sections == {
        "prompt": "You are Omega.",
        "skills": "(send content)\next rule",
        "memory_directory": "/repo/memory",
        "history": '("2026-10-28 09:00:00" "HUMAN_MESSAGE: hello")',
        "time": "2026-10-28 10:00:00",
    }


def test_parse_system_context_rejects_unknown_prompt():
    with pytest.raises(ValueError):
        mindplex.parse_system_context("You are a different prompt")


def test_unknown_system_prompt_passes_through_the_hook(clock, provider):
    providers.registerRequestProcessor(mindplex.PROCESSOR_ID, mindplex.process_request)
    original = llmRequest(llmRequestMessage("system", "custom"), [user("hi")], 50, "medium", [])

    providers.llmProviderChat(original)

    assert provider.requests == [original]


# mindplex.process_request


def test_first_turn_compiles_system_and_active_task(clock):
    history = history_record("2026-10-28 09:00:00", "earlier question")

    request = mindplex.process_request(loop_request(user("write the summary"), history=history))

    roles = [message.role for message in request.messages]
    system, context, turn = contents(request)
    assert roles == ["system", "user", "user"]
    assert "You are Omega." in system
    assert "/repo/memory" in system
    assert "2026-10-28 10:00:00" in system
    assert "earlier question" in context
    assert "CURRENT_TASK" not in context
    assert turn == "write the summary"
    assert "_newline_" not in system
    assert [tool.name for tool in request.tools] == ["send"]
    assert request.max_tokens == OUTPUT_TOKENS


def test_latest_tool_step_stays_native_and_is_not_duplicated(clock):
    mindplex.process_request(loop_request(user("research topic")))
    step = tool_step("call-1", "websearch", "result alpha", content="topic")

    request = mindplex.process_request(loop_request(*step, user(NO_INPUT)))

    assert request.messages[2:4] == step
    assert request.messages[-1].content == NO_INPUT
    assert sum("result alpha" in content for content in contents(request)) == 1
    assert "[CURRENT_TASK]\nresearch topic" in request.messages[1].content


def test_older_tool_results_are_kept_as_labeled_evidence(clock):
    mindplex.process_request(loop_request(user("research topic")))
    mindplex.process_request(
        loop_request(*tool_step("call-1", "websearch", "result alpha", content="topic"), user(NO_INPUT))
    )

    request = mindplex.process_request(
        loop_request(*tool_step("call-2", "read-file", "file beta", filename="a.txt"), user(NO_INPUT))
    )

    task = request.messages[1].content
    assert '[TOOL_CALL id=call-1] websearch {"content": "topic"}' in task
    assert "result alpha" in task
    assert "SUCCESS, RETURN:" not in task


def test_new_user_message_starts_a_new_task(clock):
    mindplex.process_request(loop_request(user("first task")))
    mindplex.process_request(
        loop_request(*tool_step("call-1", "shell", "old output", cmd="ls"), user(NO_INPUT))
    )
    mindplex.process_request(
        loop_request(*tool_step("call-2", "shell", "more output", cmd="ls"), user(NO_INPUT))
    )

    request = mindplex.process_request(loop_request(user("second task")))

    assert task_evidence.records() == ()
    assert all("old output" not in content for content in contents(request))
    assert request.messages[-1].content == "second task"


def test_idle_wake_up_starts_a_new_task_generation(clock):
    mindplex.process_request(loop_request(user("first task")))
    mindplex.process_request(
        loop_request(*tool_step("call-1", "shell", "old output", cmd="ls"), user(NO_INPUT))
    )
    generation = task_evidence.stats().task_generation

    clock.now += 601
    request = mindplex.process_request(loop_request(user(NO_INPUT)))

    assert task_evidence.stats().task_generation == generation + 1
    assert all("first task" not in content for content in contents(request))


def test_omitted_evidence_is_marked_and_can_be_recalled(clock):
    mindplex.process_request(loop_request(user("compare sources")))
    base = mindplex.process_request(loop_request(user(NO_INPUT)))
    base_tokens = sum(words(content) + 4 for content in contents(base)) + 2
    older = "older " * 100
    newer = "newer " * 100
    task_evidence.append(older, 50000)
    task_evidence.append(newer, 50000)
    mindplex.configure(
        context_window_tokens=base_tokens + OUTPUT_TOKENS + 150,
        max_output_tokens=OUTPUT_TOKENS,
        max_new_input_loops=10,
    )

    first = mindplex.process_request(loop_request(user(NO_INPUT)))

    assert "newer newer" in first.messages[1].content
    assert "older older" not in first.messages[1].content
    assert "TOOL_RESULT_OMITTED id=tool-result-1" in first.messages[1].content

    assert mindplex.recall("tool-result-1").startswith("RECALL-QUEUED")
    second = mindplex.process_request(loop_request(user(NO_INPUT)))

    assert "older older" in second.messages[1].content
    assert "newer newer" not in second.messages[1].content


def test_turn_message_object_is_passed_through_unchanged(clock):
    turn = user("Step 2026-10-28 10:00:00: what_apostrophe_s new")

    request = mindplex.process_request(loop_request(turn))

    assert request.messages[-1] is turn


def test_native_messages_over_budget_leave_request_unchanged(clock, provider):
    providers.registerRequestProcessor(mindplex.PROCESSOR_ID, mindplex.process_request)
    mindplex.process_request(loop_request(user("big task")))
    original = loop_request(*tool_step("call-1", "shell", "x " * 5000, cmd="cat"), user(NO_INPUT))

    providers.llmProviderChat(original)

    assert provider.requests == [original]


def test_manifest_reports_native_tokens(clock):
    mindplex.process_request(loop_request(user("task")))
    mindplex.process_request(
        loop_request(*tool_step("call-1", "shell", "output text", cmd="ls"), user(NO_INPUT))
    )

    manifest = mindplex._last_manifest
    assert manifest["native_message_tokens"] > 0
    assert manifest["input_token_budget"] == 2000 - OUTPUT_TOKENS - manifest["native_message_tokens"]
