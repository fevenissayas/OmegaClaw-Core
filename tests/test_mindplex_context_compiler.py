"""Interface-level tests for token-budgeted context compilation."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "plugins" / "mindplex"))

import unittest

import task_evidence as evidence
from context_compiler import (
    ContextBudgetError,
    ContextCompiler,
    ContextInput,
    ContextRecord,
    estimate_tokens,
    history_records,
    loop_context_records,
    validate_context_budget,
)


def words(text):
    return len(text.split())


def record(record_id, text, priority, required=False, message_role="system"):
    return ContextRecord(record_id, "TEST", text, priority, required, message_role)


class ContextCompilerTests(unittest.TestCase):
    def tearDown(self):
        evidence.reset()

    def test_reserves_output_tokens_and_keeps_whole_records(self):
        compiler = ContextCompiler(count_tokens=words)
        context = ContextInput(
            records=(
                record("task-facts", "required facts", 100, required=True),
                record("evidence-a", "alpha evidence complete", 80, message_role="user"),
                record("evidence-b", "beta evidence complete", 80, message_role="user"),
                record("history-a", "old conversation record", 20, message_role="user"),
            ),
            task_message="write now",
            turn_message="write now",
            context_window_tokens=31,
            max_output_tokens=8,
        )

        compiled = compiler.compile(context)

        self.assertEqual(compiled.manifest.input_token_budget, 23)
        self.assertEqual(compiled.manifest.context_window_tokens, 31)
        self.assertEqual(compiled.manifest.max_output_tokens, 8)
        self.assertEqual(compiled.manifest.included_record_ids, ("task-facts", "evidence-b"))
        self.assertEqual(compiled.manifest.omitted_record_ids, ("evidence-a", "history-a"))
        self.assertEqual(compiled.manifest.candidate_records.count, 4)
        self.assertEqual(compiled.manifest.included_records.count, 2)
        self.assertEqual(compiled.manifest.omitted_records.count, 2)
        system = compiled.request.messages[0].content
        user = compiled.request.messages[1].content
        self.assertNotIn("beta evidence complete", system)
        self.assertIn("beta evidence complete", user)
        self.assertNotIn("alpha evidence complete", user)
        self.assertNotIn("old conversation record", user)

    def test_evidence_outranks_history_even_when_history_is_newer(self):
        compiler = ContextCompiler(count_tokens=words)
        context = ContextInput(
            records=(
                record("required", "rules", 100, required=True),
                record("evidence", "source body", 80),
                record("history", "recent chatter", 20),
            ),
            task_message="compose",
            turn_message="compose",
            context_window_tokens=23,
            max_output_tokens=4,
        )

        compiled = compiler.compile(context)

        self.assertIn("evidence", compiled.manifest.included_record_ids)
        self.assertIn("history", compiled.manifest.omitted_record_ids)

    def test_recall_rank_prefers_the_first_requested_record(self):
        compiler = ContextCompiler(count_tokens=words)
        context = ContextInput(
            records=(
                ContextRecord("first", "TOOL_RESULT", "first " * 20, 80, False, "user", (), 3),
                ContextRecord("second", "TOOL_RESULT", "second " * 20, 80, False, "user", (), 2),
                ContextRecord("third", "TOOL_RESULT", "third " * 20, 80, False, "user", (), 1),
            ),
            task_message="write",
            turn_message="write",
            context_window_tokens=70,
            max_output_tokens=8,
        )

        compiled = compiler.compile(context)

        self.assertIn("first", compiled.manifest.included_record_ids)
        self.assertIn("third", compiled.manifest.omitted_record_ids)

    def test_oversized_required_record_degrades_to_an_explicit_placeholder(self):
        compiler = ContextCompiler(count_tokens=words)
        context = ContextInput(
            records=(record("oversized-task", "one two three four five six", 100, required=True),),
            task_message="one two three four five six",
            turn_message="",
            context_window_tokens=34,
            max_output_tokens=4,
        )

        compiled = compiler.compile(context)

        self.assertIn("oversized-task", compiled.manifest.omitted_record_ids)
        self.assertIn("CONTEXT_RECORD_OMITTED", compiled.request.messages[0].content)

    def test_history_reader_returns_complete_stable_records(self):
        content = (
            '("2026-08-26 10:00:00" "HUMAN_MESSAGE: A")\n'
            '("2026-08-26 10:01:00" "HUMAN_MESSAGE: B")\n'
        )
        first = history_records(content)
        second = history_records(content)

        self.assertEqual(
            [item.text for item in first],
            [
                '("2026-08-26 10:00:00" "HUMAN_MESSAGE: A")',
                '("2026-08-26 10:01:00" "HUMAN_MESSAGE: B")',
            ],
        )
        self.assertEqual([item.id for item in first], [item.id for item in second])

    def test_history_reader_ignores_parentheses_and_record_like_text_inside_strings(self):
        records = history_records(
            '("2026-08-26 10:00:00" "line one\\n(\\\"2026-08-27 fake\\\" nested)")\n'
            '("2026-08-26 10:01:00" (send "done"))\n'
        )

        self.assertEqual(len(records), 2)
        self.assertTrue(records[0].text.endswith('nested)\")'))
        self.assertEqual(records[1].text, '("2026-08-26 10:01:00" (send "done"))')

    def test_history_reader_discards_a_leading_partial_record(self):
        records = history_records(
            'truncated text (send "not a record"))\n'
            '("2026-08-26 10:01:00" (send "complete"))\n'
        )

        self.assertEqual(
            [item.text for item in records],
            ['("2026-08-26 10:01:00" (send "complete"))'],
        )

    def test_history_reader_recovers_after_unbalanced_parentheses_in_leading_fragment(self):
        complete = "".join(
            f'("2026-08-26 10:{index:02d}:00" "record {index}")\n'
            for index in range(20)
        )
        for fragment in ("partial :( emoticon", "partial NAIROBI (Reuters"):
            with self.subTest(fragment=fragment):
                self.assertEqual(len(history_records(f"{fragment}\n{complete}")), 20)

    def test_omitted_tool_result_leaves_an_explicit_placeholder(self):
        context = ContextInput(
            records=(
                record("prompt", "rules", 100, required=True),
                ContextRecord("tool-result-1", "TOOL_RESULT", "A" * 200, 80, False, "user"),
            ),
            task_message="write",
            turn_message="write",
            context_window_tokens=95,
            max_output_tokens=20,
        )

        compiled = ContextCompiler().compile(context)

        self.assertIn("tool-result-1", compiled.manifest.omitted_record_ids)
        self.assertIn(
            "[TOOL_RESULT_OMITTED id=tool-result-1 original_chars=200 "
            "recall='recall \"tool-result-1\"']",
            compiled.request.messages[1].content,
        )
        manifest = compiled.manifest.as_dict()
        self.assertEqual(manifest["tool_results"]["candidate"]["chars"], 200)
        self.assertEqual(manifest["tool_results"]["omitted"]["chars"], 200)
        self.assertLess(manifest["tool_results"]["rendered"]["chars"], 200)

    def test_source_card_is_the_only_receipt_for_hidden_chunks(self):
        source = evidence.SourceDocument(
            id="source-1",
            url="https://example.com/story",
            title="Story",
            text="complete source text",
            published_at=None,
            chunk_ids=tuple(f"source-1-chunk-{index}" for index in range(1, 8)),
        )
        chunks = tuple(
            evidence.EvidenceRecord(
                id=chunk_id,
                text=(f"chunk {index} evidence " * 120),
                kind="source_chunk",
                source_id=source.id,
                url=source.url,
                title=source.title,
                chunk_index=index,
                chunk_count=7,
            )
            for index, chunk_id in enumerate(source.chunk_ids, start=1)
        )
        adapted = loop_context_records(
            prompt="",
            skills="",
            prompt_extensions="",
            output_format="",
            memory_directory="",
            current_time="",
            evidence_records=chunks,
            history="",
            evidence_sources=(source,),
        )
        source_records = tuple(
            item for item in adapted if item.kind in {"SOURCE_CARD", "SOURCE_CHUNK"}
        )

        compiled = ContextCompiler(count_tokens=words).compile(
            ContextInput(
                records=source_records,
                task_message="write",
                turn_message="write",
                context_window_tokens=350,
                max_output_tokens=20,
            )
        )

        user = compiled.request.messages[1].content
        self.assertIn("[SOURCE_CARD id=source-1]", user)
        self.assertIn("Visible chunks: (none)", user)
        self.assertIn("Hidden chunks: source-1-chunk-1", user)
        self.assertIn('Recall hidden: recall "source-1-chunk-1,', user)
        self.assertNotIn("SOURCE_CHUNK_OMITTED", user)
        self.assertEqual(compiled.manifest.candidate_tool_results.count, 7)
        self.assertEqual(compiled.manifest.omitted_tool_results.count, 7)

    def test_five_source_batch_can_recall_three_chunks_without_losing_sources(self):
        sections = []
        bodies = []
        for index in range(1, 6):
            body = (f"source {index} evidence sentence. " * 450).rstrip()
            bodies.append(body)
            sections.append(
                f"[{index}/5] Title: Story {index}\n"
                f"URL: https://example.com/{index}\n\n"
                f"{body}"
            )
        evidence.append(
            evidence.UNTRUSTED_WEB_WARNING + "\n\n" + "\n\n".join(sections),
            50_000,
        )

        def compile_current():
            return ContextCompiler().compile(
                ContextInput(
                    records=loop_context_records(
                        prompt="writer rules",
                        skills="recall comma_separated_ids",
                        prompt_extensions="",
                        output_format="send final text",
                        memory_directory="/tmp",
                        current_time="now",
                        evidence_records=evidence.records(),
                        history="",
                        evidence_sources=evidence.sources(),
                    ),
                    task_message="write one article",
                    turn_message="continue",
                    context_window_tokens=32768,
                    max_output_tokens=16000,
                )
            )

        first = compile_current()
        self.assertLessEqual(
            first.manifest.estimated_input_tokens,
            first.manifest.input_token_budget,
        )
        self.assertGreater(first.manifest.omitted_tool_results.count, 0)
        first_user = first.request.messages[1].content
        self.assertEqual(first_user.count("[SOURCE_CARD id="), 5)
        self.assertTrue(all(body not in first_user for body in bodies))

        sources = evidence.sources()
        recalled = tuple(sources[index].chunk_ids[0] for index in (0, 2, 4))
        evidence.recall(",".join(recalled))
        evidence.append("verification search result", 50_000)
        second = compile_current()

        self.assertTrue(set(recalled).issubset(second.manifest.included_record_ids))
        self.assertEqual(len(evidence.sources()), 5)
        self.assertNotIn("SOURCE_CHUNK_OMITTED", second.request.messages[1].content)

        evidence.reset()
        self.assertEqual(evidence.records(), ())
        self.assertEqual(evidence.sources(), ())

    def test_context_budget_validation_rejects_impossible_allocations(self):
        self.assertEqual(validate_context_budget(100, 40), 60)
        for window, output in ((0, 10), (100, 0), (100, 100), (100, 101)):
            with self.subTest(window=window, output=output):
                with self.assertRaises(ContextBudgetError):
                    validate_context_budget(window, output)

    def test_history_selection_is_a_contiguous_recent_suffix(self):
        context = ContextInput(
            records=(
                record("prompt", "rules", 100, required=True),
                ContextRecord("history-old", "HISTORY_RECORD", "old", 20, False, "user"),
                ContextRecord("history-middle", "HISTORY_RECORD", "middle " * 100, 20, False, "user"),
                ContextRecord("history-new", "HISTORY_RECORD", "new", 20, False, "user"),
            ),
            task_message="continue",
            turn_message="continue",
            context_window_tokens=40,
            max_output_tokens=10,
        )

        compiled = ContextCompiler(count_tokens=words).compile(context)

        self.assertIn("history-new", compiled.manifest.included_record_ids)
        self.assertIn("history-middle", compiled.manifest.omitted_record_ids)
        self.assertIn("history-old", compiled.manifest.omitted_record_ids)

    def test_fallback_estimator_is_conservative_for_cjk(self):
        self.assertGreaterEqual(estimate_tokens("机器人" * 10), 40)

    def test_evidence_records_have_task_local_stable_ids(self):
        evidence.append("first", 100)
        evidence.append("second", 100)

        self.assertEqual(
            [(item.id, item.text) for item in evidence.records()],
            [("tool-result-1", "first"), ("tool-result-2", "second")],
        )

        evidence.reset()
        evidence.append("new task", 100)
        self.assertEqual(
            [(item.id, item.text) for item in evidence.records()],
            [("tool-result-1", "new task")],
        )


class SeparateTurnTests(unittest.TestCase):
    def compile(self, task, turn):
        return ContextCompiler(count_tokens=words).compile(
            ContextInput(
                records=(record("history", "old record", 20, message_role="user"),),
                task_message=task,
                turn_message=turn,
                context_window_tokens=200,
                max_output_tokens=10,
                separate_turn=True,
            )
        )

    def test_turn_is_its_own_last_message(self):
        compiled = self.compile("summarize the page", "summarize the page")

        self.assertEqual(
            [message.role for message in compiled.request.messages],
            ["system", "user", "user"],
        )
        self.assertIn("old record", compiled.request.messages[1].content)
        self.assertNotIn("CURRENT_TASK", compiled.request.messages[1].content)
        self.assertEqual(compiled.request.messages[2].content, "summarize the page")

    def test_continuation_turn_carries_the_task_in_context(self):
        compiled = self.compile("summarize the page", "[NO NEW USER INPUT]")

        self.assertIn("[CURRENT_TASK]\nsummarize the page", compiled.request.messages[1].content)
        self.assertEqual(compiled.request.messages[2].content, "[NO NEW USER INPUT]")

    def test_turn_tokens_count_against_the_budget(self):
        compiled = self.compile("", "one two three four five")

        self.assertEqual(compiled.manifest.estimated_input_tokens, 2 + 3 * 4 + sum(
            words(message.content) for message in compiled.request.messages
        ))


if __name__ == "__main__":
    unittest.main()
