# Mindplex context plugin

Compiles every LLM request of the Omega loop within the model's token budget,
and keeps tool results as task evidence across turns.

Upstream Omega sends the system prompt (with a 30,000-character history tail)
plus only the latest step's tool calls. Older tool results are gone after one
turn. This plugin keeps them for the whole task and lets the model recall
results that did not fit.

## How it works

The plugin registers a request processor with `providers.registerRequestProcessor`.
Before each provider call it:

1. Splits the `getContext` system prompt (`src/loop.metta`) into its sections
   and decodes `string-safe` escapes (`_newline_`, `_quote_`, `_apostrophe_`).
2. Starts a new task on a new user message, or after the loop has been idle
   for `wakeupInterval`; otherwise stores the previous step's tool results as
   evidence, labeled with the tool call that produced them.
3. Compiles prompt, skills, evidence, whole history records and time with the
   context compiler. Records are never cut mid-way; what does not fit is
   replaced by an omission marker with its id.
4. Passes the latest step's native tool-call messages through unchanged and
   reserves their tokens in the budget.

If anything fails (an unexpected system prompt, a budget that cannot hold the
required records) the hook logs the error and sends the original request.

## Skill

| Skill | Arguments | Description |
| --- | --- | --- |
| `recall` | `ids` | Queues omitted evidence by id (comma separated) for the next request. Reports queued, deferred, too-large and unknown ids. |

## Configuration

| Key | Default | Description |
| --- | --- | --- |
| `contextWindowTokens` | 32768 | Context window of the model. `maxOutputToken` is reserved from it. |
| `maxOutputToken` | 6000 | Shared with the loop. |
| `maxNewInputLoops` | 50 | Shared with the loop; used to report remaining interactions on recall. |
| `wakeupInterval` | 600 | Shared with the loop; idle time after which a new task generation starts. |
| `maxFeedback` | 50000 | Shared with the loop; legacy limit for the evidence `render` view. |

Keys are read with `config_get_by_key`, so they can be set in
`config/config.yaml`, as `OMEGA_<key>` environment variables or on the command
line.

## Logs

Each request logs `[CONTEXT_MANIFEST]` (budget, included and omitted record
ids, sizes) and `[TASK_TELEMETRY]` (task generation, interaction, evidence
counters). `[CONTEXT_BUDGET_PRESSURE]` warns at 90% of the input budget.

## Files

| File | Description |
| --- | --- |
| `mindplex.metta` | Plugin entry point; registers the `recall` skill |
| `mindplex.py` | Request processor, task tracking, recall |
| `context_compiler.py` | Token-budgeted whole-record compiler |
| `task_evidence.py` | Task evidence store, source chunking, recall queue |

Tests: `tests/test_mindplex_*.py`.
