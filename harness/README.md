# Python agent runtime

An alternative to the MeTTa turn loop (`src/loop.metta`). Python owns the
loop; MeTTa keeps doing what it is good at: skills, NAL/PLN reasoning,
memory and channels. The MeTTa loop is unchanged and stays the default.

```
channel ──receive──▶ Runner ──▶ PydanticAI Agent ──▶ model (OpenAI-compatible)
                       │              │
                       │              ├─ history processor: Dave's context compiler
                       │              └─ tools: one per MeTTa skill + recall
                       │                          │
                       ▼                          ▼
              SQLite event log            MettaBridge ──▶ PeTTa (in process)
```

## What changes compared with the MeTTa loop

| | MeTTa loop | Python runtime |
| --- | --- | --- |
| Steps | One per 1 s tick | Back to back; only waiting for input polls (0.2 s) |
| Model output | Up to 5 lines of s-expressions, repaired with `balance_parentheses` | Native tool calls with typed, named string parameters |
| Context | Dave's compiler on a text prompt | Same compiler, as a PydanticAI history processor; the latest tool step stays native |
| Task state | In memory | Saved after every step to `memory/harness.sqlite`; a crashed task resumes |
| Tool failures | Error strings in the prompt | `ERROR: ...` tool returns, recorded as events |
| Step limit | `maxNewInputLoops` loop counter | `UsageLimits(request_limit=maxNewInputLoops)` |
| Evaluation | Logs | `events` table: every tool call, result, step limit, error, resume |

## Running

The repository must be checked out as `<PeTTa>/repos/OmegaClaw-Core` (the
library name the MeTTa code imports).

```sh
pip install "pydantic-ai-slim[openai]"
cd <PeTTa>/repos/OmegaClaw-Core
OPENAI_API_KEY=... python -m harness commchannel=irc harnessModel=gpt-4.1-mini
```

`key=value` arguments are passed as `OMEGA_<key>` environment variables, so
every existing configuration key works.

| Key | Default | Description |
| --- | --- | --- |
| `harnessModel` | `gpt-4.1-mini` | Model name for the OpenAI-compatible API |
| `harnessBaseUrl` | OpenAI | e.g. `https://openrouter.ai/api/v1` |
| `harnessApiKeyEnv` | `OPENAI_API_KEY` | Environment variable holding the API key |
| `harnessToolTimeout` | 120 | Seconds before a skill call is reported as failed |
| `contextWindowTokens` | 32768 | Model window; `maxOutputToken` is reserved from it |
| `maxNewInputLoops` | 50 | Model requests per task |
| `wakeupInterval` | 600 | Idle seconds before an autonomous wake-up task |
| `maxFeedback` | 50000 | Characters kept per tool result |

## Files

| File | Description |
| --- | --- |
| `__main__.py` | Boots PeTTa, runs `(omegaclawInit)`, wires everything, serves |
| `runner.py` | Task lifecycle, per-step checkpoints, resume, wake-ups |
| `agent.py` | PydanticAI agent: skill tools, recall, event recording |
| `context.py` | Context compiler as a history processor; task evidence by tool call id |
| `metta.py` | MeTTa bridge: skill calls with escaping, timeouts, error detection |
| `skills.py` | Parses `(getSkills)` lines into typed tool specs |
| `store.py` | SQLite tasks and events |

Tests: `tests/test_harness_units.py` (no model) and
`tests/test_harness_runtime.py` (scripted model, needs `pydantic_ai`).

## Known limits

- Skill parameters are strings, as in the MeTTa catalogue.
- A skill call that times out keeps running in PeTTa (Prolog cannot be
  interrupted) and holds the bridge lock until it returns.
- Only OpenAI-compatible APIs are wired in `build_model`.
- Not yet verified inside the Docker image or against a real model.
