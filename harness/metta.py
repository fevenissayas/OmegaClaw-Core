"""Runs MeTTa skills in PeTTa for the Python runtime."""

from dataclasses import dataclass
import re
import threading


_ERROR = re.compile(r"^\(Error\b")


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: str
    error: str | None = None

    def for_model(self) -> str:
        return self.value if self.ok else f"ERROR: {self.error}"


class MettaBridge:
    """Evaluates MeTTa expressions on a PeTTa instance."""

    def __init__(self, petta, timeout_seconds: float = 120.0):
        self._petta = petta
        self._lock = threading.Lock()
        self._timeout = float(timeout_seconds)

    def eval(self, expression: str) -> list[str]:
        """Evaluates one expression and returns PeTTa's result strings."""
        with self._lock:
            return [str(result) for result in self._petta.process_metta_string(f"!{expression}")]

    def call_skill(self, name: str, args) -> ToolResult:
        expression = skill_expression(name, args)
        outcome: dict[str, object] = {}

        def run():
            try:
                outcome["results"] = self.eval(expression)
            except Exception as error:
                outcome["error"] = f"{type(error).__name__}: {error}"

        worker = threading.Thread(target=run, name=f"metta-{name}", daemon=True)
        worker.start()
        worker.join(self._timeout)
        if worker.is_alive():
            return ToolResult(False, "", f"{name} did not finish within {self._timeout:g}s")
        if "error" in outcome:
            return ToolResult(False, "", str(outcome["error"]))
        value = "\n".join(outcome.get("results", []))
        if _ERROR.match(value):
            return ToolResult(False, value, value)
        return ToolResult(True, value)

    def skill_lines(self) -> list[str]:
        """Returns the (getSkills) text lines."""
        return [line for result in self.eval("(getSkills)") for line in metta_strings(result)]


def skill_expression(name: str, args) -> str:
    """Builds a (name "arg" ...) skill call with escaped arguments."""
    if not re.fullmatch(r"[A-Za-z][\w\-]*", name):
        raise ValueError(f"Invalid skill name: {name!r}")
    quoted = " ".join(metta_quote(str(arg)) for arg in args)
    return f"({name} {quoted})" if quoted else f"({name})"


def metta_text(results) -> str:
    """Returns a result's text, unquoting a single string literal."""
    text = "\n".join(str(result) for result in results)
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        values = metta_strings(stripped)
        if len(values) == 1:
            return values[0]
    return text


def metta_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def metta_strings(text: str) -> list[str]:
    """Returns the unescaped string literals in a printed MeTTa expression."""
    values = []
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        index += 1
        value = []
        while index < len(text):
            char = text[index]
            if char == "\\" and index + 1 < len(text):
                value.append(text[index + 1])
                index += 2
            elif char == '"':
                index += 1
                break
            else:
                value.append(char)
                index += 1
        values.append("".join(value))
    return values
