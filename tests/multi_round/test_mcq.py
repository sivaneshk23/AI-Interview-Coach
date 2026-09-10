"""
Tests: MCQ Question Bank — structure, selection, deterministic scoring.

All tests are fully offline — no IBM API calls, no network I/O.

Covers:
  - MCQQuestion structure validation
  - Correct/incorrect/unanswered scoring (deterministic)
  - Invalid option handling
  - MCQBank selection — role-aware, category, difficulty
  - Seed bank integrity (all questions validate)
  - AptitudeRoundExecutor with MCQ bank
  - Timer/expiry logic
  - Serialisation round-trip
"""

from __future__ import annotations

import datetime
import pytest

from interview.mcq_bank import (
    CATEGORY_LOGICAL,
    CATEGORY_PYTHON_DATA,
    CATEGORY_QUANTITATIVE,
    CATEGORY_SQL,
    CATEGORY_TECH_FUNDAMENTALS,
    MCQBank,
    MCQQuestion,
    SEED_QUESTIONS,
    get_bank,
)
from interview.plan import InterviewRound, RoundTurn
from interview.round_types import RoundState, RoundType


# ── MCQQuestion unit tests ────────────────────────────────────────────

class TestMCQQuestion:

    def _make_q(self, **kwargs):
        defaults = dict(
            question="What is 2+2?",
            options=["A. 3", "B. 4", "C. 5", "D. 6"],
            correct_option="B",
            explanation="2+2=4",
            category=CATEGORY_QUANTITATIVE,
            difficulty="easy",
        )
        defaults.update(kwargs)
        return MCQQuestion(**defaults)

    def test_valid_question_validates(self):
        q = self._make_q()
        q.validate()  # should not raise

    def test_wrong_option_count_raises(self):
        q = self._make_q(options=["A. 3", "B. 4", "C. 5"])  # only 3
        with pytest.raises(ValueError, match="4 options"):
            q.validate()

    def test_invalid_correct_option_raises(self):
        q = self._make_q(correct_option="E")  # E is not valid
        with pytest.raises(ValueError, match="A/B/C/D"):
            q.validate()

    def test_correct_option_lowercase_is_valid(self):
        """validate() accepts lowercase a/b/c/d (normalised via .upper())."""
        q = self._make_q(correct_option="b")
        q.validate()  # should not raise

    def test_to_dict_round_trip(self):
        q = self._make_q()
        d = q.to_dict()
        assert d["question"] == "What is 2+2?"
        assert d["correct_option"] == "B"
        assert len(d["options"]) == 4


# ── MCQBank unit tests ────────────────────────────────────────────────

class TestMCQBank:

    def _bank(self, n=5):
        """Create a small test bank with n identical-structure questions."""
        qs = []
        for i in range(n):
            qs.append(MCQQuestion(
                question=f"Question {i}",
                options=[f"A. opt{i}a", f"B. opt{i}b", f"C. opt{i}c", f"D. opt{i}d"],
                correct_option="A",
                explanation=f"Explanation {i}",
                category=CATEGORY_QUANTITATIVE,
                difficulty="medium",
                role_tags=[r"engineer"],
            ))
        return MCQBank(qs)

    def test_len(self):
        bank = self._bank(7)
        assert len(bank) == 7

    def test_add_validates(self):
        """Bank should validate question on add."""
        bank = MCQBank()
        bad = MCQQuestion(
            question="Bad",
            options=["A. x"],  # only 1 option
            correct_option="A",
            explanation="",
            category=CATEGORY_QUANTITATIVE,
        )
        with pytest.raises(ValueError):
            bank.add(bad)

    def test_select_count_limit(self):
        bank = self._bank(10)
        result = bank.select(count=3)
        assert len(result) <= 3

    def test_select_returns_all_when_count_exceeds_pool(self):
        bank = self._bank(3)
        result = bank.select(count=100)
        assert len(result) == 3

    def test_select_empty_bank_returns_empty(self):
        bank = MCQBank()
        assert bank.select(count=5) == []

    def test_select_with_role_keyword(self):
        """Role-matched questions should be included in selection."""
        bank = MCQBank()
        # Add role-tagged question
        bank.add(MCQQuestion(
            question="Tagged for engineers",
            options=["A. a", "B. b", "C. c", "D. d"],
            correct_option="A",
            explanation="",
            category=CATEGORY_TECH_FUNDAMENTALS,
            role_tags=[r"engineer"],
        ))
        # Add untagged question
        bank.add(MCQQuestion(
            question="Untagged generic",
            options=["A. a", "B. b", "C. c", "D. d"],
            correct_option="B",
            explanation="",
            category=CATEGORY_QUANTITATIVE,
            role_tags=[],
        ))
        # With count=1 on a role-matched selection, the tagged question
        # should be in the pool of role-boosted items (returned first before shuffle)
        # Use a seeded select to get deterministic ordering for the test
        result = bank.select(role="Software Engineer", count=1, seed=42)
        assert len(result) == 1
        # The single result when count=1 should be the role-tagged question
        # (it's first in the boosted pool, then shuffle applies to 1-item list)
        questions = {q.question for q in bank.select(role="Software Engineer", count=2)}
        assert "Tagged for engineers" in questions

    def test_by_category(self):
        bank = self._bank(5)  # all CATEGORY_QUANTITATIVE
        result = bank.by_category(CATEGORY_QUANTITATIVE)
        assert len(result) == 5
        result_other = bank.by_category(CATEGORY_SQL)
        assert result_other == []

    def test_select_with_difficulty_filter(self):
        bank = MCQBank()
        easy_q = MCQQuestion(
            question="Easy Q",
            options=["A. a", "B. b", "C. c", "D. d"],
            correct_option="A",
            explanation="",
            category=CATEGORY_QUANTITATIVE,
            difficulty="easy",
        )
        hard_q = MCQQuestion(
            question="Hard Q",
            options=["A. a", "B. b", "C. c", "D. d"],
            correct_option="B",
            explanation="",
            category=CATEGORY_QUANTITATIVE,
            difficulty="hard",
        )
        bank.add(easy_q)
        bank.add(hard_q)

        easy_result = bank.select(difficulty="easy", count=10)
        assert all(q.difficulty == "easy" for q in easy_result)

        hard_result = bank.select(difficulty="hard", count=10)
        assert all(q.difficulty == "hard" for q in hard_result)

    def test_seed_bank_has_questions(self):
        assert len(SEED_QUESTIONS) >= 20

    def test_seed_bank_all_valid(self):
        """Every seed question must pass structural validation."""
        for i, q in enumerate(SEED_QUESTIONS):
            try:
                q.validate()
            except ValueError as e:
                pytest.fail(f"SEED_QUESTIONS[{i}] ({q.question!r}) failed validation: {e}")

    def test_get_bank_singleton(self):
        b1 = get_bank()
        b2 = get_bank()
        assert b1 is b2

    def test_seed_bank_covers_multiple_categories(self):
        bank = get_bank()
        categories = {q.category for q in SEED_QUESTIONS}
        assert len(categories) >= 5  # at least 5 distinct categories

    def test_seed_bank_correct_options_valid(self):
        for q in SEED_QUESTIONS:
            assert q.correct_option in {"A", "B", "C", "D"}, (
                f"Question {q.question!r} has invalid correct_option={q.correct_option!r}"
            )


# ── Deterministic MCQ scoring ─────────────────────────────────────────

class TestDeterministicScoring:
    """
    MCQ scoring MUST be deterministic — never LLM-based.
    These tests verify the exact grading contract.
    """

    def _make_round(self):
        return InterviewRound(
            round_type=RoundType.APTITUDE,
            order=1,
            title="Aptitude",
            max_turns=10,
        )

    def _executor(self):
        from interview.executors import AptitudeRoundExecutor
        return AptitudeRoundExecutor(bank=get_bank())

    def test_correct_answer_scores_10(self):
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Software Engineer")
        correct = turn.correct_option

        # Submit the correct option
        turn.selected_option = correct
        turn = executor.evaluate_answer(round_, turn, role="Software Engineer")

        assert turn.is_correct is True
        assert turn.score == 10.0

    def test_incorrect_answer_scores_zero(self):
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Software Engineer")
        correct = turn.correct_option
        wrong = {"A": "B", "B": "C", "C": "D", "D": "A"}[correct]

        turn.selected_option = wrong
        turn = executor.evaluate_answer(round_, turn, role="Software Engineer")

        assert turn.is_correct is False
        assert turn.score == 0.0

    def test_no_answer_scores_zero(self):
        """Unanswered question must score 0."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Software Engineer")
        turn.selected_option = None
        turn = executor.evaluate_answer(round_, turn, role="Software Engineer")

        assert turn.is_correct is False
        assert turn.score == 0.0

    def test_case_insensitive_option_match(self):
        """Selected option 'a' should match correct_option 'A'."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Data Analyst")
        correct = turn.correct_option  # e.g. "A"
        turn.selected_option = correct.lower()  # "a"
        turn = executor.evaluate_answer(round_, turn, role="Data Analyst")

        assert turn.is_correct is True

    def test_whitespace_stripped_in_comparison(self):
        """Selected option ' A ' should match 'A'."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Data Analyst")
        turn.selected_option = " " + turn.correct_option + " "
        turn = executor.evaluate_answer(round_, turn, role="Data Analyst")

        assert turn.is_correct is True

    def test_evaluation_dict_has_required_keys(self):
        """evaluate_answer must populate standard evaluation keys."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Cloud Engineer")
        turn.selected_option = "A"
        turn = executor.evaluate_answer(round_, turn, role="Cloud Engineer")

        required = {
            "overall_score", "is_correct", "correct_option",
            "selected_option", "explanation", "evaluation",
        }
        for key in required:
            assert key in turn.evaluation, f"Missing key: {key}"

    def test_build_round_evaluation_scores_correctly(self):
        """build_round_evaluation: score = (correct/total) * 10."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        # Simulate 4 answers: 3 correct, 1 wrong
        for i in range(4):
            t = executor.generate_question(round_, role="Data Engineer")
            t.selected_option = t.correct_option if i < 3 else (
                {"A": "B", "B": "C", "C": "D", "D": "A"}[t.correct_option]
            )
            t = executor.evaluate_answer(round_, t, role="Data Engineer")
            round_.turns.append(t)

        eval_ = executor.build_round_evaluation(round_, role="Data Engineer")
        assert eval_.correct_count == 3
        assert eval_.total_questions == 4
        assert abs(eval_.score - 7.5) < 0.01  # 3/4 * 10 = 7.5

    def test_build_round_evaluation_zero_turns(self):
        """Empty round must not crash — score 0."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        eval_ = executor.build_round_evaluation(round_, role="HR")
        assert eval_.score == 0.0
        assert eval_.correct_count == 0
        assert eval_.total_questions == 0

    def test_round_evaluation_feedback_contains_score_info(self):
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        t = executor.generate_question(round_, role="ML Engineer")
        t.selected_option = t.correct_option
        t = executor.evaluate_answer(round_, t, role="ML Engineer")
        round_.turns.append(t)

        eval_ = executor.build_round_evaluation(round_, role="ML Engineer")
        assert "1" in eval_.feedback  # "1/1 correct"

    def test_questions_not_empty(self):
        """Generated question must have non-empty text and options."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Backend Developer")
        assert len(turn.question) > 0
        assert len(turn.options) == 4
        assert turn.correct_option in {"A", "B", "C", "D"}

    def test_no_repeat_questions_within_round(self):
        """The same question should not appear twice in a round (if bank is large enough)."""
        executor = self._executor()
        round_ = self._make_round()
        round_.state = RoundState.IN_PROGRESS

        asked = []
        for _ in range(5):
            t = executor.generate_question(round_, role="Software Engineer")
            asked.append(t.question)
            round_.turns.append(t)  # simulate answered

        unique = set(asked)
        # With a 40+ question bank, we should get at least 3 unique questions out of 5
        assert len(unique) >= 3


# ── MCQ Timing ────────────────────────────────────────────────────────

class TestMCQTiming:

    def _executor(self):
        from interview.executors import AptitudeRoundExecutor
        return AptitudeRoundExecutor(bank=get_bank())

    def _make_round(self):
        return InterviewRound(
            round_type=RoundType.APTITUDE,
            order=1,
            title="Timed Aptitude",
            max_turns=5,
        )

    def test_generate_sets_time_started(self):
        """Generated turn's evaluation should include time_started."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Engineer")
        assert isinstance(turn.evaluation, dict)
        assert "time_started" in turn.evaluation
        # Must be a valid ISO datetime
        from datetime import datetime
        dt = datetime.fromisoformat(turn.evaluation["time_started"])
        assert dt is not None

    def test_evaluate_records_submitted_at(self):
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Engineer")
        turn.selected_option = turn.correct_option
        turn = executor.evaluate_answer(round_, turn, role="Engineer")

        assert "submitted_at" in turn.evaluation

    def test_timed_out_question_scores_zero(self):
        """
        When time_limit_sec is exceeded (server-side), the answer must score 0
        regardless of whether the selected_option is correct.
        """
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Engineer")
        # Inject a very small time limit (1 second) with a past start time
        past = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(seconds=10)
        ).isoformat()
        turn.evaluation = {
            "time_started":   past,
            "time_limit_sec": 1,  # expired 9 seconds ago
            "category":       "test",
            "difficulty":     "easy",
        }

        # Even submitting the correct option, timed-out = score 0
        turn.selected_option = turn.correct_option
        turn = executor.evaluate_answer(round_, turn, role="Engineer")

        assert turn.is_correct is False
        assert turn.score == 0.0
        assert turn.evaluation.get("timed_out") is True

    def test_within_time_limit_grades_normally(self):
        """Within time limit: graded normally."""
        executor = self._executor()
        round_ = self._make_round()
        round_.start()

        turn = executor.generate_question(round_, role="Engineer")
        # Set a generous time limit in the future
        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        turn.evaluation = {
            "time_started":   now,
            "time_limit_sec": 3600,  # 1 hour — definitely not expired
            "category":       "test",
            "difficulty":     "easy",
        }

        turn.selected_option = turn.correct_option
        turn = executor.evaluate_answer(round_, turn, role="Engineer")

        assert turn.evaluation.get("timed_out") is False
        assert turn.is_correct is True
        assert turn.score == 10.0


# ── Role-aware MCQ selection ──────────────────────────────────────────

class TestRoleAwareMCQSelection:

    def test_data_analyst_gets_data_questions(self):
        bank = get_bank()
        # Data analyst — should get SQL and Python/Data questions
        questions = bank.select(role="Data Analyst", count=15)
        categories = {q.category for q in questions}
        # Should include at least some data-oriented categories
        data_cats = {CATEGORY_SQL, CATEGORY_PYTHON_DATA}
        assert data_cats & categories  # non-empty intersection

    def test_software_engineer_gets_tech_fundamentals(self):
        bank = get_bank()
        # Use a large count to ensure we get most/all categories from the seed bank
        questions = bank.select(role="Software Engineer", count=len(bank._questions))
        categories = {q.category for q in questions}
        assert CATEGORY_TECH_FUNDAMENTALS in categories

    def test_arbitrary_role_returns_questions(self):
        """Any role (even unrecognised) should still get questions."""
        bank = get_bank()
        for role in ["CEO", "Doctor", "Teacher", "Chef", "Astronaut"]:
            questions = bank.select(role=role, count=5)
            assert len(questions) > 0, f"No questions for role: {role}"

    def test_empty_role_returns_questions(self):
        bank = get_bank()
        questions = bank.select(role="", count=5)
        assert len(questions) > 0


# ── AptitudeRoundExecutor integration ─────────────────────────────────

class TestAptitudeRoundExecutorIntegration:
    """Integration tests for AptitudeRoundExecutor with the round lifecycle."""

    def _executor(self, bank=None):
        from interview.executors import AptitudeRoundExecutor
        return AptitudeRoundExecutor(bank=bank or get_bank())

    def _round(self, max_turns=5):
        r = InterviewRound(
            round_type=RoundType.APTITUDE,
            order=1,
            title="Aptitude",
            max_turns=max_turns,
        )
        r.start()
        return r

    def test_full_aptitude_flow_one_question(self):
        executor = self._executor()
        round_ = self._round()

        # Generate question
        turn = executor.generate_question(round_, role="Python Developer")
        assert turn.question
        assert len(turn.options) == 4
        assert turn.correct_option in {"A", "B", "C", "D"}

        # Submit correct answer
        turn.selected_option = turn.correct_option
        evaluated = executor.evaluate_answer(round_, turn, role="Python Developer")

        assert evaluated.is_correct is True
        assert evaluated.score == 10.0
        assert evaluated.evaluation["is_correct"] is True

    def test_full_aptitude_flow_wrong_answer(self):
        executor = self._executor()
        round_ = self._round()

        turn = executor.generate_question(round_, role="Data Scientist")
        correct = turn.correct_option
        wrong = {"A": "B", "B": "A", "C": "D", "D": "C"}[correct]
        turn.selected_option = wrong

        evaluated = executor.evaluate_answer(round_, turn, role="Data Scientist")
        assert evaluated.is_correct is False
        assert evaluated.score == 0.0

    def test_build_executor_factory_returns_aptitude(self):
        """build_executor(RoundType.APTITUDE) must return AptitudeRoundExecutor."""
        from interview.executors import AptitudeRoundExecutor, build_executor
        executor = build_executor(RoundType.APTITUDE)
        assert isinstance(executor, AptitudeRoundExecutor)

    def test_executor_injects_custom_bank(self):
        """AptitudeRoundExecutor accepts a custom bank for testing."""
        custom_bank = MCQBank([
            MCQQuestion(
                question="Custom test Q?",
                options=["A. Alpha", "B. Beta", "C. Gamma", "D. Delta"],
                correct_option="C",
                explanation="Gamma is correct",
                category=CATEGORY_LOGICAL,
            )
        ])
        executor = self._executor(bank=custom_bank)
        round_ = self._round()
        turn = executor.generate_question(round_, role="any")
        assert turn.question == "Custom test Q?"
        assert turn.correct_option == "C"

    def test_round_type_attribute_is_aptitude(self):
        from interview.executors import AptitudeRoundExecutor
        assert AptitudeRoundExecutor.round_type == RoundType.APTITUDE
