"""Typed tool specs parsed from the MeTTa skill catalogue.

getSkills (src/skills.metta) returns one text line per skill, written as
"- <description>: <name> <arg> <arg>", plus free-text rule lines. The
runtime turns each skill line into a SkillSpec so the model gets one tool
per skill with named string parameters; rule lines stay prompt text.
"""

from dataclasses import dataclass
import re


_SKILL_LINE = re.compile(r"^-\s+(?P<description>.+):\s+(?P<call>[A-Za-z][\w\-]*(?:\s+[\w\-]+)*)\s*$")
_TOOL_NAME = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    params: tuple[str, ...]

    @property
    def tool_name(self) -> str:
        """Tool name accepted by LLM APIs (letters, digits, _ and -)."""
        return _TOOL_NAME.sub("_", self.name)


def parse_skill_lines(lines) -> tuple[tuple[SkillSpec, ...], tuple[str, ...]]:
    """Split getSkills lines into skill specs and remaining rule lines."""
    specs: dict[str, SkillSpec] = {}
    rules: list[str] = []
    for raw in lines:
        line = str(raw).strip()
        match = _SKILL_LINE.match(line)
        if not match:
            if line:
                rules.append(line)
            continue
        name, *args = match.group("call").split()
        params = tuple(_unique_params(args))
        # A later definition replaces an earlier one, like dynamic skills do.
        specs.pop(name, None)
        specs[name] = SkillSpec(name, match.group("description").strip(), params)
    return tuple(specs.values()), tuple(rules)


def _unique_params(args):
    seen: dict[str, int] = {}
    for arg in args:
        base = arg.replace("-", "_")
        count = seen.get(base, 0) + 1
        seen[base] = count
        yield base if count == 1 else f"{base}_{count}"
