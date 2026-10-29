"""Tests skill parsing, the MeTTa bridge and the event store."""

import threading

import pytest

from harness.metta import (
    MettaBridge,
    metta_quote,
    metta_strings,
    metta_text,
    skill_expression,
)
from harness.skills import SkillSpec, parse_skill_lines
from harness.store import DONE, RUNNING, EventStore


STATIC_SKILLS = [
    "- Remember a particular string such as skills and memories: remember string",
    "- Episodes searches history for episodes around a time stamp, time format is same as in TIME: episodes time_string",
    "- Write string to file, the result is read back from disk (bytes, sha256, head/tail) - relay it, never claim a write succeeded without it: write-file filename string",
    "- Get a list of allowed base paths for reading, writing, and updating files: get-io-policy",
    "- Execute MeTTa expression: metta sexpression",
    "Example to invoke Non-Axiomatic Logic via MeTTa: ",
    "metta (|- ((--> (× sam garfield) friend) (stv 1.0 0.9))",
]




def test_parse_skill_lines_builds_typed_specs_and_keeps_rules():
    specs, rules = parse_skill_lines(STATIC_SKILLS)

    assert specs == (
        SkillSpec("remember", "Remember a particular string such as skills and memories", ("string",)),
        SkillSpec(
            "episodes",
            "Episodes searches history for episodes around a time stamp, time format is same as in TIME",
            ("time_string",),
        ),
        SkillSpec(
            "write-file",
            "Write string to file, the result is read back from disk (bytes, sha256, head/tail) "
            "- relay it, never claim a write succeeded without it",
            ("filename", "string"),
        ),
        SkillSpec(
            "get-io-policy",
            "Get a list of allowed base paths for reading, writing, and updating files",
            (),
        ),
        SkillSpec("metta", "Execute MeTTa expression", ("sexpression",)),
    )
    assert rules == (
        "Example to invoke Non-Axiomatic Logic via MeTTa:",
        "metta (|- ((--> (× sam garfield) friend) (stv 1.0 0.9))",
    )


def test_later_skill_definition_replaces_earlier_one():
    specs, _ = parse_skill_lines(["- Old: send string", "- New sender: send message"])

    assert specs == (SkillSpec("send", "New sender", ("message",)),)


def test_tool_name_and_repeated_params_are_api_safe():
    specs, _ = parse_skill_lines(["- Copy: copy-file path path"])

    assert specs[0].tool_name == "copy-file"
    assert specs[0].params == ("path", "path_2")




def test_skill_expression_quotes_and_escapes_arguments():
    assert skill_expression("send", ['say "hi" \\ bye']) == '(send "say \\"hi\\" \\\\ bye")'
    assert skill_expression("version", []) == "(version)"


def test_skill_expression_rejects_injected_names():
    with pytest.raises(ValueError):
        skill_expression("send) (shell", ["rm"])


def test_metta_strings_round_trip_quote():
    text = 'line "one"\nback\\slash'

    assert metta_strings(metta_quote(text)) == [text]


def test_metta_text_unquotes_a_single_string_result():
    assert metta_text(['"hello \\"you\\""']) == 'hello "you"'
    assert metta_text(["(a b)"]) == "(a b)"
    assert metta_text(['"a" "b"']) == '"a" "b"'


class FakePetta:
    def __init__(self, results=None, error=None, block=None):
        self.results = results or []
        self.error = error
        self.block = block
        self.expressions = []

    def process_metta_string(self, expression):
        self.expressions.append(expression)
        if self.block is not None:
            self.block.wait()
        if self.error is not None:
            raise self.error
        return self.results


def test_call_skill_evaluates_expression_and_returns_value():
    petta = FakePetta(results=["SEND-SUCCESS"])

    result = MettaBridge(petta).call_skill("send", ["hi"])

    assert petta.expressions == ['!(send "hi")']
    assert result.ok and result.value == "SEND-SUCCESS"


def test_call_skill_reports_metta_error_results():
    result = MettaBridge(FakePetta(results=["(Error (shell) failed)"])).call_skill("shell", ["x"])

    assert not result.ok
    assert result.for_model() == "ERROR: (Error (shell) failed)"


def test_call_skill_reports_exceptions():
    result = MettaBridge(FakePetta(error=RuntimeError("janus"))).call_skill("shell", ["x"])

    assert not result.ok
    assert result.error == "RuntimeError: janus"


def test_call_skill_reports_timeout():
    block = threading.Event()
    bridge = MettaBridge(FakePetta(block=block), timeout_seconds=0.05)

    result = bridge.call_skill("shell", ["sleep"])
    block.set()

    assert not result.ok
    assert "did not finish within 0.05s" in result.error


def test_skill_lines_reads_get_skills_strings():
    petta = FakePetta(results=['("- Send: send string" "rule line")'])

    assert MettaBridge(petta).skill_lines() == ["- Send: send string", "rule line"]




def test_store_tracks_task_lifecycle_and_events(tmp_path):
    store = EventStore(str(tmp_path / "harness.sqlite"))
    task_id = store.start_task("summarize")
    store.record(task_id, "tool_result", call_id="c1", name="send", result="ok")
    store.save_history(task_id, b"[]")

    assert store.interrupted_task().id == task_id
    assert store.interrupted_task().status == RUNNING

    store.finish_task(task_id, DONE)

    assert store.interrupted_task() is None
    assert store.task(task_id).history == b"[]"
    assert [(e["kind"], e["name"]) for e in store.events(task_id)] == [("tool_result", "send")]


def test_store_survives_reopening(tmp_path):
    path = str(tmp_path / "harness.sqlite")
    first = EventStore(path)
    task_id = first.start_task("long task")
    first.save_history(task_id, b"saved")
    first.close()

    second = EventStore(path)

    assert second.interrupted_task().history == b"saved"
