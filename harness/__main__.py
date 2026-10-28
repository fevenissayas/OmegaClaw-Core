"""Run OmegaClaw with the Python runtime: python -m harness [key=value ...]

PeTTa loads lib_omegaclaw and runs (omegaclawInit) (config, memory,
plugins, channels), then the Python runner owns the loop: it receives
messages through the MeTTa channel layer and runs MeTTa skills as tools.
key=value arguments become OMEGA_<key> environment variables, the same
override config_get_by_key already reads.
"""

import logging
import os
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[1]
LIBRARY_NAME = "OmegaClaw-Core"
BOOT = (
    "!(import! &self (library lib_import))",
    "!(git-import! \"https://github.com/asi-alliance/OmegaClaw-Core.git\")",
    f"!(import! &self (library {LIBRARY_NAME} lib_omegaclaw))",
    "!(omegaclawInit)",
)


def main(argv: list[str]) -> int:
    for argument in argv:
        key, separator, value = argument.partition("=")
        if not separator:
            print(f"Expected key=value, got {argument!r}", file=sys.stderr)
            return 2
        os.environ[f"OMEGA_{key}"] = value

    if REPO.name != LIBRARY_NAME or REPO.parent.name != "repos":
        print(
            f"The repository must be checked out as <PeTTa>/repos/{LIBRARY_NAME}, "
            f"found {REPO}",
            file=sys.stderr,
        )
        return 2
    petta_path = pathlib.Path(os.environ.get("PETTA_PATH", REPO.parents[1]))
    for path in (REPO, REPO / "src", petta_path / "python"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from petta import PeTTa
    from config import config_get_by_key

    from .agent import build_agent
    from .context import ContextProcessor, PromptSections, TaskEvidence
    from .metta import MettaBridge, metta_quote, metta_text
    from .runner import Runner
    from .skills import parse_skill_lines
    from .store import EventStore

    logging.basicConfig(level=logging.INFO)
    os.chdir(petta_path)
    petta = PeTTa(petta_path=str(petta_path))
    for expression in BOOT:
        petta.process_metta_string(expression)
    bridge = MettaBridge(petta, timeout_seconds=float(config_get_by_key("harnessToolTimeout", 120)))

    skills, rules = parse_skill_lines(bridge.skill_lines())
    memory_directory = metta_text(bridge.eval("(memoryDirectory)"))

    def sections() -> PromptSections:
        return PromptSections(
            prompt=metta_text(bridge.eval("(getPrompt (provider))")),
            rules="\n".join(rules),
            prompt_extensions=metta_text(bridge.eval("(getPromptExtensions)")),
            memory_directory=memory_directory,
            history=metta_text(bridge.eval("(getHistory)")),
            current_time=metta_text(bridge.eval("(get_time_as_string)")),
        )

    def append_history(message, tool_results, output):
        calls = " ".join(f"({event['name']})" for event in tool_results)
        record = " ".join(
            metta_quote(part)
            for part in (
                metta_text(bridge.eval("(get_time_as_string)")),
                f"HUMAN_MESSAGE: {message}",
                f"TOOLS: {calls}",
                f"OUTPUT: {output}",
            )
        )
        with open(pathlib.Path(memory_directory) / "history.metta", "a", encoding="utf-8") as history:
            history.write(f"({record})\n")

    max_output = int(config_get_by_key("maxOutputToken", 6000))
    max_feedback = int(config_get_by_key("maxFeedback", 50000))
    task_evidence = TaskEvidence(max_feedback)
    processor = ContextProcessor(
        sections,
        task_evidence,
        context_window_tokens=int(config_get_by_key("contextWindowTokens", 32768)),
        max_output_tokens=max_output,
    )
    agent = build_agent(
        build_model(config_get_by_key),
        skills,
        bridge.call_skill,
        processor,
        max_output_tokens=max_output,
    )
    store = EventStore(str(pathlib.Path(memory_directory) / "harness.sqlite"))
    runner = Runner(
        agent,
        store,
        processor,
        task_evidence,
        max_steps=int(config_get_by_key("maxNewInputLoops", 50)),
        max_feedback_chars=max_feedback,
        append_history=append_history,
    )
    runner.serve(
        lambda: metta_text(bridge.eval("(receive)")),
        wakeup_seconds=float(config_get_by_key("wakeupInterval", 600)),
    )
    return 0


def build_model(config_get_by_key):
    """OpenAI-compatible chat model: OpenAI, OpenRouter, ASI Cloud, local servers."""
    from pydantic_ai.providers.openai import OpenAIProvider

    try:
        from pydantic_ai.models.openai import OpenAIChatModel as ChatModel
    except ImportError:  # pydantic-ai before 1.0
        from pydantic_ai.models.openai import OpenAIModel as ChatModel

    api_key = os.environ.get(str(config_get_by_key("harnessApiKeyEnv", "OPENAI_API_KEY")))
    base_url = config_get_by_key("harnessBaseUrl", "") or None
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return ChatModel(str(config_get_by_key("harnessModel", "gpt-4.1-mini")), provider=provider)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
