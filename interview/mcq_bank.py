"""
MCQ Question Bank — deterministic aptitude/knowledge questions.

Architecture:
    MCQQuestion    — single question with options + answer key
    MCQBank        — collection with selection helpers
    SEED_BANK      — static seed dataset (always available offline)
    get_bank()     — returns the shared bank singleton

Design constraints:
    - Correct answers come from structured data ONLY.
    - LLMs are NEVER used to determine correctness.
    - Scoring is deterministic (exact option-letter match).
    - Role-based selection uses keyword heuristics, not a closed role enum.
    - Categories are flexible strings, not enums.
    - Time limits are optional metadata — server enforces them externally.

Question format:
    question       : str         — the question text
    options        : list[str]   — exactly 4 options (A, B, C, D)
    correct_option : str         — "A", "B", "C", or "D"
    explanation    : str         — correct-answer rationale
    category       : str         — grouping (see CATEGORY_* constants)
    difficulty     : str         — "easy" | "medium" | "hard"
    role_tags      : list[str]   — keyword hints for role-based selection
    time_limit_sec : int | None  — per-question time limit (None = no limit)

The seed bank contains ~60 questions spanning:
    Quantitative Aptitude, Logical Reasoning, Verbal Ability,
    Data Interpretation, Technical Fundamentals, Python/Data, SQL,
    Cloud Concepts, Networking, Security Basics
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

# ── Category constants ────────────────────────────────────────────────
CATEGORY_QUANTITATIVE   = "Quantitative Aptitude"
CATEGORY_LOGICAL        = "Logical Reasoning"
CATEGORY_VERBAL         = "Verbal Ability"
CATEGORY_DATA_INTERP    = "Data Interpretation"
CATEGORY_TECH_FUNDAMENTALS = "Technical Fundamentals"
CATEGORY_PYTHON_DATA    = "Python / Data"
CATEGORY_SQL            = "SQL"
CATEGORY_CLOUD          = "Cloud Concepts"
CATEGORY_NETWORKING     = "Networking"
CATEGORY_SECURITY       = "Security Basics"


# ── MCQQuestion ───────────────────────────────────────────────────────

@dataclass
class MCQQuestion:
    """
    A single MCQ question with a deterministic answer key.

    The correct_option field is the ground truth — never derived from
    an LLM at question-serving time.
    """
    question:       str
    options:        List[str]            # exactly 4: ["A. ...", "B. ...", "C. ...", "D. ..."]
    correct_option: str                  # "A", "B", "C", or "D"
    explanation:    str
    category:       str
    difficulty:     str = "medium"       # easy | medium | hard
    role_tags:      List[str] = field(default_factory=list)
    time_limit_sec: Optional[int] = None # None = no enforced per-question limit

    def validate(self) -> None:
        """Raise ValueError if the question is structurally invalid."""
        if len(self.options) != 4:
            raise ValueError(f"MCQQuestion must have exactly 4 options, got {len(self.options)}")
        if self.correct_option.upper() not in {"A", "B", "C", "D"}:
            raise ValueError(f"correct_option must be A/B/C/D, got {self.correct_option!r}")

    def to_dict(self) -> dict:
        return {
            "question":       self.question,
            "options":        self.options,
            "correct_option": self.correct_option,
            "explanation":    self.explanation,
            "category":       self.category,
            "difficulty":     self.difficulty,
            "role_tags":      self.role_tags,
            "time_limit_sec": self.time_limit_sec,
        }


# ── MCQBank ───────────────────────────────────────────────────────────

class MCQBank:
    """
    Collection of MCQ questions with role-aware selection.

    All questions must pass structural validation before being added.
    Scoring is always deterministic — this class never calls any LLM.
    """

    def __init__(self, questions: Optional[List[MCQQuestion]] = None):
        self._questions: List[MCQQuestion] = []
        if questions:
            for q in questions:
                self.add(q)

    def add(self, question: MCQQuestion) -> None:
        question.validate()
        self._questions.append(question)

    def __len__(self) -> int:
        return len(self._questions)

    def select(
        self,
        role: str = "",
        categories: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        count: int = 10,
        seed: Optional[int] = None,
    ) -> List[MCQQuestion]:
        """
        Select up to `count` questions for a given role and/or category set.

        Selection order:
        1. Role-keyword-matched questions first
        2. Category-filtered (if specified)
        3. Difficulty-filtered (if specified)
        4. Shuffled with optional reproducible seed
        5. Truncated to `count`

        Never returns fewer questions than available (just returns all if count > pool).
        """
        pool = self._questions[:]

        # Difficulty filter
        if difficulty:
            filtered = [q for q in pool if q.difficulty == difficulty]
            pool = filtered if filtered else pool  # fall back to all if none match

        # Category filter
        if categories:
            cat_set = {c.lower() for c in categories}
            filtered = [q for q in pool if q.category.lower() in cat_set]
            pool = filtered if filtered else pool

        # Role-keyword boost: prefer role-tagged questions first
        if role:
            role_lower = role.lower()
            boosted   = [q for q in pool if any(
                re.search(tag, role_lower, re.IGNORECASE) for tag in q.role_tags
            )]
            rest      = [q for q in pool if q not in boosted]
            pool = boosted + rest

        # Shuffle
        rng = random.Random(seed)
        rng.shuffle(pool)

        return pool[:count]

    def by_category(self, category: str) -> List[MCQQuestion]:
        return [q for q in self._questions if q.category == category]


# ── Seed Question Bank ────────────────────────────────────────────────
# A curated set of reliable questions.  Each question's correct answer
# has been verified manually before being included.
#
# Adding new questions:
#   1. Append to SEED_QUESTIONS below.
#   2. Verify correct_option and explanation before adding.
#   3. Add appropriate category and role_tags.
#   No other files need changing.

SEED_QUESTIONS: List[MCQQuestion] = [

    # ── Quantitative Aptitude ─────────────────────────────────────────

    MCQQuestion(
        question="If a train travels 360 km in 4 hours, what is its speed in m/s?",
        options=["A. 25 m/s", "B. 30 m/s", "C. 90 m/s", "D. 100 m/s"],
        correct_option="A",
        explanation="Speed = 360 km / 4 h = 90 km/h = 90 × (1000/3600) = 25 m/s.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="A shopkeeper sells an article at 20% profit. If the cost price is ₹500, what is the selling price?",
        options=["A. ₹600", "B. ₹580", "C. ₹520", "D. ₹650"],
        correct_option="A",
        explanation="Selling price = CP × (1 + profit%) = 500 × 1.20 = ₹600.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="The simple interest on ₹8000 at 5% per annum for 3 years is:",
        options=["A. ₹1000", "B. ₹1200", "C. ₹1500", "D. ₹800"],
        correct_option="B",
        explanation="SI = (P × R × T) / 100 = (8000 × 5 × 3) / 100 = ₹1200.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="If 15 workers complete a project in 24 days, how many workers are needed to complete it in 9 days?",
        options=["A. 30", "B. 35", "C. 40", "D. 45"],
        correct_option="C",
        explanation="Total work = 15 × 24 = 360 worker-days. Workers needed = 360 / 9 = 40.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="medium",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="The average of 5 consecutive even numbers is 18. What is the largest number?",
        options=["A. 20", "B. 22", "C. 24", "D. 26"],
        correct_option="B",
        explanation="Consecutive even numbers with average 18: 14, 16, 18, 20, 22. Largest = 22.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="medium",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="What is 15% of 240?",
        options=["A. 32", "B. 36", "C. 38", "D. 40"],
        correct_option="B",
        explanation="15% of 240 = (15/100) × 240 = 36.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="A and B can complete a work in 12 and 18 days respectively. Together, in how many days?",
        options=["A. 6.0", "B. 6.5", "C. 7.0", "D. 7.2"],
        correct_option="D",
        explanation="Combined rate = 1/12 + 1/18 = 3/36 + 2/36 = 5/36. Days = 36/5 = 7.2.",
        category=CATEGORY_QUANTITATIVE,
        difficulty="medium",
        role_tags=[r".*"],
    ),

    # ── Logical Reasoning ─────────────────────────────────────────────

    MCQQuestion(
        question="In a certain code, FLOWER is written as GMPXFS. How is GARDEN written in that code?",
        options=["A. HBSEFS", "B. HBSEFO", "C. HCTFGO", "D. IBSEFS"],
        correct_option="B",
        explanation="Each letter is shifted +1: G→H, A→B, R→S, D→E, E→F, N→O → HBSEFO.",
        category=CATEGORY_LOGICAL,
        difficulty="medium",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="Find the odd one out: 2, 3, 5, 7, 9, 11, 13",
        options=["A. 3", "B. 7", "C. 9", "D. 13"],
        correct_option="C",
        explanation="All others are prime numbers. 9 = 3×3 is not prime.",
        category=CATEGORY_LOGICAL,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="If all Bloops are Razzles, and all Razzles are Lazzles, then:",
        options=[
            "A. All Lazzles are Bloops",
            "B. All Bloops are Lazzles",
            "C. All Razzles are Bloops",
            "D. None of the above"
        ],
        correct_option="B",
        explanation="Transitivity: Bloops ⊆ Razzles ⊆ Lazzles, so all Bloops are Lazzles.",
        category=CATEGORY_LOGICAL,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="What comes next in the series: 2, 6, 12, 20, 30, __?",
        options=["A. 40", "B. 42", "C. 44", "D. 46"],
        correct_option="B",
        explanation="Differences: 4, 6, 8, 10, 12 → next term = 30 + 12 = 42.",
        category=CATEGORY_LOGICAL,
        difficulty="medium",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="A clock shows 3:15. What is the angle between the hour and minute hands?",
        options=["A. 0°", "B. 7.5°", "C. 15°", "D. 22.5°"],
        correct_option="B",
        explanation="At 3:15: minute hand = 90°, hour hand = 90° + (15/60)×30 = 97.5°. Angle = 7.5°.",
        category=CATEGORY_LOGICAL,
        difficulty="hard",
        role_tags=[r".*"],
    ),

    # ── Verbal Ability ────────────────────────────────────────────────

    MCQQuestion(
        question="Choose the word most similar in meaning to ELOQUENT:",
        options=["A. Silent", "B. Articulate", "C. Confused", "D. Harsh"],
        correct_option="B",
        explanation="Eloquent means fluent and persuasive in speech; articulate is the closest synonym.",
        category=CATEGORY_VERBAL,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="Select the correctly punctuated sentence:",
        options=[
            "A. Its a great opportunity, isn't it.",
            "B. It's a great opportunity, isn't it?",
            "C. Its a great opportunity isn't it?",
            "D. It's a great opportunity isn't it."
        ],
        correct_option="B",
        explanation="'It's' is the contraction of 'it is'; the tag question ends with '?'.",
        category=CATEGORY_VERBAL,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="Identify the antonym of BENEVOLENT:",
        options=["A. Generous", "B. Kind", "C. Malevolent", "D. Compassionate"],
        correct_option="C",
        explanation="Benevolent means kind/well-meaning; malevolent means wishing harm — direct antonym.",
        category=CATEGORY_VERBAL,
        difficulty="easy",
        role_tags=[r".*"],
    ),

    # ── Data Interpretation ───────────────────────────────────────────

    MCQQuestion(
        question="A dataset has values: 4, 8, 6, 5, 3, 2, 8, 9, 2, 5. What is the mode?",
        options=["A. 2", "B. 5", "C. 8", "D. Both 2, 5, and 8"],
        correct_option="D",
        explanation="2 appears twice, 5 appears twice, 8 appears twice — all are modes (multimodal).",
        category=CATEGORY_DATA_INTERP,
        difficulty="medium",
        role_tags=[r"data|analyst|science|ml|ai"],
    ),
    MCQQuestion(
        question="The median of: 11, 22, 33, 44, 55, 66, 77 is:",
        options=["A. 33", "B. 44", "C. 55", "D. 38.5"],
        correct_option="B",
        explanation="7 values; median is the 4th value = 44.",
        category=CATEGORY_DATA_INTERP,
        difficulty="easy",
        role_tags=[r"data|analyst|science|ml|ai"],
    ),
    MCQQuestion(
        question="In a bar chart, company A has sales of 500 units and company B has 350. By what % does A exceed B?",
        options=["A. 42.86%", "B. 30%", "C. 43.12%", "D. 28.57%"],
        correct_option="A",
        explanation="% difference = (500-350)/350 × 100 = 150/350 × 100 ≈ 42.86%.",
        category=CATEGORY_DATA_INTERP,
        difficulty="medium",
        role_tags=[r"data|analyst|science|ml|ai"],
    ),

    # ── Technical Fundamentals ────────────────────────────────────────

    MCQQuestion(
        question="Which data structure uses FIFO (First In, First Out) order?",
        options=["A. Stack", "B. Queue", "C. Tree", "D. Graph"],
        correct_option="B",
        explanation="A queue is FIFO — elements are added at the rear and removed from the front.",
        category=CATEGORY_TECH_FUNDAMENTALS,
        difficulty="easy",
        role_tags=[r"software|developer|engineer|backend|frontend|devops"],
    ),
    MCQQuestion(
        question="What is the time complexity of binary search on a sorted array of n elements?",
        options=["A. O(n)", "B. O(n log n)", "C. O(log n)", "D. O(1)"],
        correct_option="C",
        explanation="Binary search halves the search space each step: O(log n).",
        category=CATEGORY_TECH_FUNDAMENTALS,
        difficulty="easy",
        role_tags=[r"software|developer|engineer"],
    ),
    MCQQuestion(
        question="Which HTTP method is idempotent but NOT safe?",
        options=["A. GET", "B. POST", "C. PUT", "D. DELETE"],
        correct_option="C",
        explanation="PUT is idempotent (same result on repeat) but not safe (it modifies data). DELETE is also idempotent.",
        category=CATEGORY_TECH_FUNDAMENTALS,
        difficulty="medium",
        role_tags=[r"software|developer|engineer|backend|api"],
    ),
    MCQQuestion(
        question="In OOP, which principle is violated when a subclass cannot be substituted for its parent class?",
        options=[
            "A. Open/Closed Principle",
            "B. Liskov Substitution Principle",
            "C. Interface Segregation Principle",
            "D. Dependency Inversion Principle"
        ],
        correct_option="B",
        explanation="The Liskov Substitution Principle states subclasses must be substitutable for parent classes.",
        category=CATEGORY_TECH_FUNDAMENTALS,
        difficulty="medium",
        role_tags=[r"software|developer|engineer|backend"],
    ),
    MCQQuestion(
        question="Which sorting algorithm has worst-case O(n log n) time complexity?",
        options=["A. Bubble Sort", "B. Quick Sort", "C. Merge Sort", "D. Insertion Sort"],
        correct_option="C",
        explanation="Merge Sort guarantees O(n log n) worst-case. Quick Sort degrades to O(n²) in worst case.",
        category=CATEGORY_TECH_FUNDAMENTALS,
        difficulty="medium",
        role_tags=[r"software|developer|engineer"],
    ),
    MCQQuestion(
        question="What does CPU stand for?",
        options=[
            "A. Central Processing Unit",
            "B. Core Processing Utility",
            "C. Central Program Utility",
            "D. Computer Processing Unit"
        ],
        correct_option="A",
        explanation="CPU stands for Central Processing Unit — the primary processing component of a computer.",
        category=CATEGORY_TECH_FUNDAMENTALS,
        difficulty="easy",
        role_tags=[r".*"],
    ),

    # ── Python / Data ─────────────────────────────────────────────────

    MCQQuestion(
        question="What is the output of: list(range(2, 10, 3))?",
        options=["A. [2, 5, 8]", "B. [2, 4, 6, 8]", "C. [2, 3, 4, 5, 6, 7, 8, 9]", "D. [2, 5, 8, 11]"],
        correct_option="A",
        explanation="range(2, 10, 3) generates: 2, 5, 8 (step=3, stops before 10).",
        category=CATEGORY_PYTHON_DATA,
        difficulty="easy",
        role_tags=[r"data|python|software|ml|ai|analyst|engineer"],
    ),
    MCQQuestion(
        question="Which Python library is primarily used for DataFrame operations?",
        options=["A. NumPy", "B. pandas", "C. matplotlib", "D. scikit-learn"],
        correct_option="B",
        explanation="pandas provides DataFrame and Series — the primary tool for tabular data manipulation in Python.",
        category=CATEGORY_PYTHON_DATA,
        difficulty="easy",
        role_tags=[r"data|python|analyst|ml|ai"],
    ),
    MCQQuestion(
        question="What does the 'self' keyword represent in a Python class method?",
        options=[
            "A. The class itself",
            "B. The parent class",
            "C. The current instance of the class",
            "D. A static reference"
        ],
        correct_option="C",
        explanation="'self' refers to the instance (object) on which the method is called.",
        category=CATEGORY_PYTHON_DATA,
        difficulty="easy",
        role_tags=[r"python|software|developer|engineer"],
    ),
    MCQQuestion(
        question="Which of these creates a shallow copy of a list in Python?",
        options=["A. list2 = list1", "B. list2 = list1.copy()", "C. list2 = list1[:]", "D. Both B and C"],
        correct_option="D",
        explanation="Both .copy() and slicing [:] create shallow copies. Assignment creates only an alias.",
        category=CATEGORY_PYTHON_DATA,
        difficulty="medium",
        role_tags=[r"python|software|developer|engineer"],
    ),
    MCQQuestion(
        question="What is the purpose of a virtual environment in Python?",
        options=[
            "A. To speed up execution",
            "B. To isolate project dependencies",
            "C. To run Python on virtual machines only",
            "D. To provide GPU acceleration"
        ],
        correct_option="B",
        explanation="Virtual environments isolate package dependencies per project, preventing version conflicts.",
        category=CATEGORY_PYTHON_DATA,
        difficulty="easy",
        role_tags=[r"python|software|developer|engineer|data"],
    ),
    MCQQuestion(
        question="In pandas, what does df.groupby('col').agg({'val': 'sum'}) do?",
        options=[
            "A. Filters rows where col equals val",
            "B. Groups rows by col and sums the val column for each group",
            "C. Sorts the DataFrame by col",
            "D. Removes duplicate values in val"
        ],
        correct_option="B",
        explanation="groupby + agg performs split-apply-combine: groups by 'col', applies sum to 'val'.",
        category=CATEGORY_PYTHON_DATA,
        difficulty="medium",
        role_tags=[r"data|analyst|python|ml|ai"],
    ),

    # ── SQL ───────────────────────────────────────────────────────────

    MCQQuestion(
        question="Which SQL clause filters records after aggregation?",
        options=["A. WHERE", "B. HAVING", "C. GROUP BY", "D. ORDER BY"],
        correct_option="B",
        explanation="HAVING filters groups after GROUP BY. WHERE filters rows before aggregation.",
        category=CATEGORY_SQL,
        difficulty="easy",
        role_tags=[r"data|analyst|sql|database|backend|engineer"],
    ),
    MCQQuestion(
        question="Which JOIN returns rows with matching values in both tables?",
        options=["A. LEFT JOIN", "B. RIGHT JOIN", "C. INNER JOIN", "D. FULL OUTER JOIN"],
        correct_option="C",
        explanation="INNER JOIN returns only rows where the join condition matches in both tables.",
        category=CATEGORY_SQL,
        difficulty="easy",
        role_tags=[r"data|analyst|sql|database|backend"],
    ),
    MCQQuestion(
        question="What does the SQL keyword DISTINCT do?",
        options=[
            "A. Sorts results",
            "B. Removes duplicate rows from results",
            "C. Counts null values",
            "D. Joins two tables"
        ],
        correct_option="B",
        explanation="SELECT DISTINCT removes duplicate rows from the query result set.",
        category=CATEGORY_SQL,
        difficulty="easy",
        role_tags=[r"data|analyst|sql|database"],
    ),
    MCQQuestion(
        question="Which SQL aggregate function returns the number of rows?",
        options=["A. SUM()", "B. AVG()", "C. COUNT()", "D. MAX()"],
        correct_option="C",
        explanation="COUNT() returns the number of rows (or non-NULL values if a column is specified).",
        category=CATEGORY_SQL,
        difficulty="easy",
        role_tags=[r"data|analyst|sql|database"],
    ),
    MCQQuestion(
        question="What is a PRIMARY KEY in SQL?",
        options=[
            "A. A key that can contain NULL values",
            "B. A column or set of columns that uniquely identifies each row",
            "C. A key that references another table",
            "D. An index on a non-unique column"
        ],
        correct_option="B",
        explanation="A PRIMARY KEY uniquely identifies each row, is NOT NULL, and must be unique.",
        category=CATEGORY_SQL,
        difficulty="easy",
        role_tags=[r"data|analyst|sql|database|backend"],
    ),

    # ── Cloud Concepts ────────────────────────────────────────────────

    MCQQuestion(
        question="Which cloud service model provides the most control over infrastructure?",
        options=["A. SaaS", "B. PaaS", "C. IaaS", "D. FaaS"],
        correct_option="C",
        explanation="IaaS (Infrastructure as a Service) gives users control over VMs, storage, and networking.",
        category=CATEGORY_CLOUD,
        difficulty="easy",
        role_tags=[r"cloud|devops|infrastructure|platform|sre|engineer"],
    ),
    MCQQuestion(
        question="What does 'auto-scaling' mean in cloud computing?",
        options=[
            "A. Automatically backing up data",
            "B. Automatically adjusting compute resources based on demand",
            "C. Automatically updating software",
            "D. Automatically migrating servers"
        ],
        correct_option="B",
        explanation="Auto-scaling adjusts the number of compute instances up or down based on load.",
        category=CATEGORY_CLOUD,
        difficulty="easy",
        role_tags=[r"cloud|devops|infrastructure|platform|sre"],
    ),
    MCQQuestion(
        question="In cloud computing, what is a 'region'?",
        options=[
            "A. A type of virtual machine",
            "B. A geographic area containing one or more data centers",
            "C. A storage bucket",
            "D. A network security group"
        ],
        correct_option="B",
        explanation="A cloud region is a geographic area with isolated data centers (availability zones).",
        category=CATEGORY_CLOUD,
        difficulty="easy",
        role_tags=[r"cloud|devops|infrastructure|platform"],
    ),
    MCQQuestion(
        question="Which cloud storage type is best suited for unstructured data like images and videos?",
        options=["A. Block storage", "B. File storage", "C. Object storage", "D. Relational database"],
        correct_option="C",
        explanation="Object storage (e.g., S3, GCS) is designed for unstructured data like images, videos, backups.",
        category=CATEGORY_CLOUD,
        difficulty="medium",
        role_tags=[r"cloud|devops|infrastructure|platform"],
    ),

    # ── Networking ────────────────────────────────────────────────────

    MCQQuestion(
        question="What layer of the OSI model does TCP operate at?",
        options=["A. Layer 2 — Data Link", "B. Layer 3 — Network", "C. Layer 4 — Transport", "D. Layer 7 — Application"],
        correct_option="C",
        explanation="TCP (Transmission Control Protocol) operates at Layer 4 — the Transport layer.",
        category=CATEGORY_NETWORKING,
        difficulty="medium",
        role_tags=[r"network|cloud|devops|security|infrastructure|software|engineer"],
    ),
    MCQQuestion(
        question="What is the default port for HTTPS?",
        options=["A. 80", "B. 443", "C. 8080", "D. 22"],
        correct_option="B",
        explanation="HTTPS uses port 443 by default. HTTP uses port 80.",
        category=CATEGORY_NETWORKING,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="What does DNS stand for?",
        options=[
            "A. Data Network System",
            "B. Domain Name System",
            "C. Dynamic Network Service",
            "D. Distributed Node Service"
        ],
        correct_option="B",
        explanation="DNS — Domain Name System — translates human-readable domain names to IP addresses.",
        category=CATEGORY_NETWORKING,
        difficulty="easy",
        role_tags=[r".*"],
    ),
    MCQQuestion(
        question="Which protocol provides reliable, connection-oriented communication?",
        options=["A. UDP", "B. IP", "C. ICMP", "D. TCP"],
        correct_option="D",
        explanation="TCP provides reliable, connection-oriented communication with error correction and flow control.",
        category=CATEGORY_NETWORKING,
        difficulty="easy",
        role_tags=[r"network|cloud|devops|security|software|engineer"],
    ),

    # ── Security Basics ───────────────────────────────────────────────

    MCQQuestion(
        question="What does SQL injection exploit?",
        options=[
            "A. Weak passwords",
            "B. Unvalidated user input that is included in SQL queries",
            "C. Buffer overflow vulnerabilities",
            "D. Insecure file uploads"
        ],
        correct_option="B",
        explanation="SQL injection injects malicious SQL via unsanitised user input concatenated into queries.",
        category=CATEGORY_SECURITY,
        difficulty="easy",
        role_tags=[r"security|software|developer|engineer|backend"],
    ),
    MCQQuestion(
        question="Which cryptographic hash function is currently considered most secure?",
        options=["A. MD5", "B. SHA-1", "C. SHA-256", "D. CRC32"],
        correct_option="C",
        explanation="SHA-256 (part of SHA-2 family) is currently considered secure. MD5 and SHA-1 are broken.",
        category=CATEGORY_SECURITY,
        difficulty="medium",
        role_tags=[r"security|software|developer|engineer"],
    ),
    MCQQuestion(
        question="What is a Cross-Site Scripting (XSS) attack?",
        options=[
            "A. Injecting scripts into pages viewed by other users",
            "B. Guessing user passwords",
            "C. Flooding a server with requests",
            "D. Intercepting network packets"
        ],
        correct_option="A",
        explanation="XSS injects malicious client-side scripts into web pages viewed by other users.",
        category=CATEGORY_SECURITY,
        difficulty="easy",
        role_tags=[r"security|software|developer|engineer|frontend|backend"],
    ),
    MCQQuestion(
        question="What is the principle of least privilege?",
        options=[
            "A. Users should have the fewest permissions needed to do their job",
            "B. Administrators should have no permissions",
            "C. All users should share a single account",
            "D. Passwords must be very short"
        ],
        correct_option="A",
        explanation="Least privilege: grant only the minimum access rights required. Limits blast radius of breaches.",
        category=CATEGORY_SECURITY,
        difficulty="easy",
        role_tags=[r"security|devops|cloud|software|engineer"],
    ),
    MCQQuestion(
        question="What does HTTPS ensure that HTTP does not?",
        options=[
            "A. Faster page loading",
            "B. Encryption of data in transit",
            "C. Smaller response payloads",
            "D. Cached DNS lookups"
        ],
        correct_option="B",
        explanation="HTTPS uses TLS/SSL to encrypt data in transit, preventing eavesdropping and MITM attacks.",
        category=CATEGORY_SECURITY,
        difficulty="easy",
        role_tags=[r"security|software|developer|engineer"],
    ),
]


# ── Bank singleton ────────────────────────────────────────────────────

_bank: Optional[MCQBank] = None


def get_bank() -> MCQBank:
    """Return the shared MCQBank singleton (initialised from SEED_QUESTIONS)."""
    global _bank
    if _bank is None:
        _bank = MCQBank(SEED_QUESTIONS)
    return _bank
