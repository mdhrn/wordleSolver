"""
wordle_engine.py
================================================================================
Headless Wordle simulation + solver engine.

Design goals (matches the AI course rubric's "informed search + Bayesian
reasoning, applied to a real problem" framing from our planning discussion):

    1. Deterministic, vectorized feedback computation — the state-transition
       function of the search problem.
    2. BaselineGreedyEntropySolver   — 1-ply informed search (Lecture 3):
       greedily picks the guess that maximizes expected information gain
       (Shannon entropy of the resulting partition of the candidate set).
    3. MultiStepLookaheadSolver      — extends (2) with a bounded 2-ply
       expectation: among the top-K first-guess candidates by 1-ply entropy,
       estimate the *expected number of candidates remaining after a second
       guess*, and pick the guess that minimizes that quantity. This is the
       standard, tractable approximation used by practical Wordle solvers to
       approach (without fully reproducing) Selby's exhaustive-search optimum
       of 3.4212 average guesses.
    4. InformationMetricsMatrix      — the runtime, 3Blue1Brown-style table:
       for any candidate set + shortlist of guesses, reports prior entropy,
       expected information, and (once an actual secret/pattern is known)
       the *actual* information received, mirroring the "bits gained" bar
       chart from the reference video.
    5. A headless benchmark harness that plays every word in the remaining
       candidate pool as the secret and reports the guess-count distribution
       — the actual number you'll quote in the report and compare against
       the 3.4212 published optimum.

Data dependencies (resolved automatically, see `resolve_paths()`):
    Prefers ./cleaned/remaining_candidates.txt and ./cleaned/full_guess_vocab.txt
    (written by the earlier eda_wordle.py cleaning pass). Falls back to
    recomputing them from the three raw files in the working directory if the
    cleaned/ folder isn't present.

Performance note ("pure Python" + numpy):
    The algorithmic logic (search, heuristics, control flow) is plain Python;
    numpy is used only as a vectorized array backend for the O(vocab x
    candidates) feedback-pattern computation, which is the actual
    performance-critical inner loop. Nothing here calls out to a compiled
    Wordle-solver library — the entropy/lookahead logic is implemented from
    scratch so you can explain every line of it.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

GRAY, YELLOW, GREEN = 0, 1, 2
PATTERN_BASE = np.array([81, 27, 9, 3, 1], dtype=np.int64)  # base-3 encoding, 5 digits
N_PATTERNS = 3 ** 5  # 243


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================



def resolve_paths(root: str | Path = ".") -> dict[str, Path]:
    """Find the word-list files, preferring the cleaned/ outputs from the
    earlier EDA pass and falling back to the raw files in `root`."""
    root = Path(root)
    cleaned = root / "cleaned"
    paths = {
        "remaining_candidates": cleaned / "remaining_candidates.txt",
        "full_guess_vocab": cleaned / "full_guess_vocab.txt",
        "raw_answers": root / "wordle-answers-alphabetical.txt",
        "raw_guesses": root / "wordle-allowed-guesses.txt",
        "raw_past": root / "past_answers.txt",
    }
    return paths


def _read_words(path: Path) -> list[str]:
    return [w.strip().lower() for w in path.read_text().splitlines() if w.strip()]


def load_word_pools(root: str | Path = ".") -> tuple[list[str], list[str]]:
    """Returns (remaining_candidates, full_guess_vocab).

    remaining_candidates: the ~473-word pool of words that COULD still be a
        future secret (past NYT answers already subtracted out).
    full_guess_vocab: every legal guess (answers ∪ allowed-guesses-only,
        ~12,972 words) — used as the search space for *guesses*, which is
        larger than the space of possible *secrets*.
    """
    paths = resolve_paths(root)
    if paths["remaining_candidates"].exists() and paths["full_guess_vocab"].exists():
        candidates = _read_words(paths["remaining_candidates"])
        guess_vocab = _read_words(paths["full_guess_vocab"])
        return candidates, guess_vocab

    # Fallback: recompute directly from the raw files (same logic as eda_wordle.py)
    answers = set(_read_words(paths["raw_answers"]))
    guesses_only = set(_read_words(paths["raw_guesses"]))
    past = set()
    for raw in paths["raw_past"].read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts:
            past.add(parts[0].lower())
    candidates = sorted(answers - past)
    guess_vocab = sorted(answers | guesses_only)
    return candidates, guess_vocab


# ==============================================================================
# 2. VECTORIZED FEEDBACK / PATTERN COMPUTATION
# ==============================================================================

def words_to_array(words: list[str]) -> np.ndarray:
    """['crane', ...] -> int8 array of shape (N, 5), values 0-25 (a=0 .. z=25)."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    arr = arr.reshape(len(words), 5).astype(np.int16) - ord("a")
    return arr


def feedback_batch(guess_idx: np.ndarray, answers_idx: np.ndarray) -> np.ndarray:
    """Compute the Wordle feedback pattern of ONE guess against MANY answers.

    guess_idx:   shape (5,)   int array, one word
    answers_idx: shape (M, 5) int array, M candidate secrets
    returns:     shape (M,)   int array, base-3-encoded pattern in [0, 243)

    Correctly handles duplicate letters using Wordle's actual rule: greens are
    resolved first, then yellows are assigned left-to-right up to however many
    unconsumed copies of that letter remain in the answer.
    """
    M = answers_idx.shape[0]
    colors = np.zeros((M, 5), dtype=np.int8)

    green_mask = answers_idx == guess_idx[None, :]
    colors[green_mask] = GREEN

    # Letters available for "yellow" = answer letters not already claimed by a green
    ans_pool = np.where(green_mask, -1, answers_idx)

    for pos in range(5):
        if green_mask[:, pos].all():
            continue  # this guess position is green for every candidate — skip
        letter = guess_idx[pos]
        is_green_here = green_mask[:, pos]
        avail = (ans_pool == letter).sum(axis=1)  # remaining copies of `letter` per answer
        can_yellow = (~is_green_here) & (avail > 0)
        colors[can_yellow, pos] = YELLOW
        # consume one copy of `letter` from the pool for rows that just went yellow
        first_match = np.argmax(ans_pool == letter, axis=1)
        consume_rows = np.where(can_yellow)[0]
        if consume_rows.size:
            ans_pool[consume_rows, first_match[consume_rows]] = -1

    return colors @ PATTERN_BASE


def pattern_to_str(pattern: int) -> str:
    """243-space integer -> human-readable 'GYX--' style string (G/Y/gray-dot)."""
    digits = []
    p = pattern
    for base in PATTERN_BASE:
        d, p = divmod(p, int(base))
        digits.append(d)
    return "".join({GREEN: "G", YELLOW: "Y", GRAY: "."}[d] for d in digits)


# ==============================================================================
# 3. THE GAME STATE / ENGINE
# ==============================================================================

@dataclass
class WordleEngine:
    """Headless engine: owns the word pools and precomputed integer encodings.
    Does not print anything or ask for input — safe to run thousands of times
    in a benchmark loop."""

    candidates: list[str]
    guess_vocab: list[str]
    _cand_arr: np.ndarray = field(init=False, repr=False)
    _guess_arr: np.ndarray = field(init=False, repr=False)
    _word_to_idx: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._cand_arr = words_to_array(self.candidates)
        self._guess_arr = words_to_array(self.guess_vocab)
        self._word_to_idx = {w: i for i, w in enumerate(self.guess_vocab)}

    def feedback(self, guess: str, secret: str) -> int:
        g = words_to_array([guess])[0]
        a = words_to_array([secret])
        return int(feedback_batch(g, a)[0])

    def partition(self, guess: str, candidate_words: list[str]) -> dict[int, list[str]]:
        """Split `candidate_words` into buckets keyed by the feedback pattern
        `guess` would produce against each of them."""
        g = words_to_array([guess])[0]
        a = words_to_array(candidate_words)
        patterns = feedback_batch(g, a)
        buckets: dict[int, list[str]] = defaultdict(list)
        for w, p in zip(candidate_words, patterns):
            buckets[int(p)].append(w)
        return buckets

    def entropy(self, guess: str, candidate_words: list[str]) -> float:
        """Expected information gain (bits) of `guess` against the current
        candidate set — the core informed-search heuristic."""
        buckets = self.partition(guess, candidate_words)
        n = len(candidate_words)
        return -sum((len(b) / n) * math.log2(len(b) / n) for b in buckets.values())


# ==============================================================================
# 4. SOLVERS
# ==============================================================================

class BaselineGreedyEntropySolver:
    """1-ply informed search: at every turn, pick the guess (from `guess_pool`)
    that maximizes expected information gain against the current candidate set.
    This is the direct implementation of the "informed search" heuristic from
    the course (Lecture 3), specialized with an entropy-based evaluation
    function instead of a distance-to-goal estimate."""

    name = "Baseline Greedy Entropy"

    def __init__(self, engine: WordleEngine, guess_pool: Optional[list[str]] = None):
        self.engine = engine
        self.guess_pool = guess_pool or engine.guess_vocab

    def choose(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError(
                "No candidates remain consistent with the feedback so far. This means "
                "either the secret isn't in the engine's candidate pool (e.g. it's an "
                "already-used past answer) or there's a feedback-consistency bug upstream."
            )
        if len(candidates) == 1:
            return candidates[0]
        best_word, best_score = None, -1.0
        for g in self.guess_pool:
            score = self.engine.entropy(g, candidates)
            # tie-break: prefer a guess that could itself be the answer
            if score > best_score or (score == best_score and g in candidates and best_word not in candidates):
                best_word, best_score = g, score
        return best_word


class MultiStepLookaheadSolver:
    """2-ply informed search. Among the top-K guesses ranked by 1-ply entropy,
    score each by the EXPECTED NUMBER OF CANDIDATES REMAINING after a second
    guess (lower is better), where the second guess is itself chosen greedily
    (by entropy) from within each resulting bucket. This is a standard,
    tractable approximation of full expectiminimax search — exact 2-ply
    expectiminimax over the full guess vocabulary is what Selby's exhaustive
    solver does to reach 3.4212 average guesses, but it requires search-tree
    pruning/caching well beyond this course's scope; this bounded version
    captures the same idea (look past the immediate guess) at a fraction of
    the compute cost.

    Falls back to brute-force exact search when the candidate set is small
    (<= exact_threshold), since at that size trying every remaining candidate
    as the next guess is cheap and exact.
    """

    name = "Multi-Step Lookahead"

    def __init__(
        self,
        engine: WordleEngine,
        guess_pool: Optional[list[str]] = None,
        top_k: int = 12,
        exact_threshold: int = 12,
    ):
        self.engine = engine
        self.guess_pool = guess_pool or engine.guess_vocab
        self.top_k = top_k
        self.exact_threshold = exact_threshold

    def _expected_remaining_after_one_more_guess(self, candidates: list[str]) -> float:
        """Best achievable expected bucket size using one further greedy-entropy
        guess drawn from `candidates` itself (cheap: only scans the shrinking
        bucket, not the full vocab)."""
        n = len(candidates)
        if n <= 1:
            return 0.0
        best = n  # worst case: no information at all
        for g in candidates:
            buckets = self.engine.partition(g, candidates)
            expected_size = sum((len(b) / n) * len(b) for b in buckets.values())
            best = min(best, expected_size)
        return best

    def choose(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError(
                "No candidates remain consistent with the feedback so far. This means "
                "either the secret isn't in the engine's candidate pool (e.g. it's an "
                "already-used past answer) or there's a feedback-consistency bug upstream."
            )
        n = len(candidates)
        if n == 1:
            return candidates[0]

        if n <= self.exact_threshold:
            # Exact-ish: try every remaining candidate as the next guess and
            # pick the one minimizing expected candidates left after it.
            scored = []
            for g in candidates:
                buckets = self.engine.partition(g, candidates)
                expected_size = sum((len(b) / n) * len(b) for b in buckets.values())
                scored.append((expected_size, g not in candidates, g))
            scored.sort(key=lambda t: (t[0], t[2]))
            return scored[0][2]

        # Stage 1: rank the whole guess pool by 1-ply entropy, keep the top K.
        entropies = [(self.engine.entropy(g, candidates), g) for g in self.guess_pool]
        entropies.sort(key=lambda t: -t[0])
        shortlist = [g for _, g in entropies[: self.top_k]]

        # Stage 2: for each shortlisted guess, estimate expected candidates
        # remaining after ONE MORE guess.
        best_word, best_score = None, math.inf
        for g in shortlist:
            buckets = self.engine.partition(g, candidates)
            exp_after_two = sum(
                (len(b) / n) * self._expected_remaining_after_one_more_guess(b)
                for b in buckets.values()
            )
            if exp_after_two < best_score:
                best_word, best_score = g, exp_after_two
        return best_word


# ==============================================================================
# 5. INFORMATION METRICS MATRIX (3Blue1Brown-style runtime table)
# ==============================================================================

class InformationMetricsMatrix:
    """Produces the running "how much information did we actually get" table.

    For a shortlist of guesses against the current candidate set, reports:
        prior_entropy_bits   -- log2(|candidates|), uncertainty before guessing
        expected_info_bits   -- E[entropy] the guess is predicted to yield
        max_bucket_frac      -- size of the largest resulting bucket / total
                                 (a cheap worst-case indicator alongside entropy)

    If `secret` is supplied, also computes the ACTUAL information received:
        actual_pattern       -- the real feedback string
        actual_info_bits     -- -log2(P(actual_pattern)) : Shannon's own
                                 definition of "how surprising was this
                                 outcome", the number 3Blue1Brown plots live
                                 as each guess resolves.
    """

    def __init__(self, engine: WordleEngine):
        self.engine = engine

    def table(
        self, candidates: list[str], guesses: list[str], secret: Optional[str] = None
    ) -> pd.DataFrame:
        n = len(candidates)
        prior_bits = math.log2(n) if n > 1 else 0.0
        rows = []
        for g in guesses:
            buckets = self.engine.partition(g, candidates)
            probs = [len(b) / n for b in buckets.values()]
            expected_info = -sum(p * math.log2(p) for p in probs)
            max_bucket_frac = max(probs)
            row = {
                "guess": g,
                "prior_entropy_bits": round(prior_bits, 3),
                "expected_info_bits": round(expected_info, 3),
                "n_buckets": len(buckets),
                "max_bucket_frac": round(max_bucket_frac, 3),
            }
            if secret is not None:
                pattern = self.engine.feedback(g, secret)
                bucket_size = len(buckets[pattern])
                p_actual = bucket_size / n
                actual_info = -math.log2(p_actual) if p_actual > 0 else float("inf")
                row["actual_pattern"] = pattern_to_str(pattern)
                row["actual_info_bits"] = round(actual_info, 3)
                row["candidates_remaining"] = bucket_size
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("expected_info_bits", ascending=False)
        return df.reset_index(drop=True)


# ==============================================================================
# 6. HEADLESS SIMULATION / BENCHMARK
# ==============================================================================

Solver = Callable[[list[str]], str]


def play_game(
    engine: WordleEngine,
    solver,
    secret: str,
    first_guess: Optional[str] = None,
    max_guesses: int = 6,
    require_valid_secret: bool = True,
) -> dict:
    """Play one full game headlessly. Returns a trace dict — no printing,
    safe for a tight benchmark loop.

    If `require_valid_secret`, raises early (instead of silently degrading)
    when `secret` isn't in `engine.candidates` — e.g. a word that's already
    been used as a real NYT answer and therefore can never come up again."""
    if require_valid_secret and secret not in engine.candidates:
        raise ValueError(
            f"'{secret}' is not in the engine's candidate pool — it may already be a "
            f"used past answer. Pass require_valid_secret=False to force it anyway."
        )
    candidates = list(engine.candidates)
    guesses_made: list[str] = []
    patterns_seen: list[str] = []

    for turn in range(1, max_guesses + 1):
        guess = first_guess if (turn == 1 and first_guess) else solver.choose(candidates)
        pattern = engine.feedback(guess, secret)
        guesses_made.append(guess)
        patterns_seen.append(pattern_to_str(pattern))

        if guess == secret:
            return {
                "secret": secret,
                "guesses": guesses_made,
                "patterns": patterns_seen,
                "n_guesses": turn,
                "solved": True,
            }
        candidates = [w for w in candidates if engine.feedback(guess, w) == pattern]

    return {
        "secret": secret,
        "guesses": guesses_made,
        "patterns": patterns_seen,
        "n_guesses": max_guesses,
        "solved": False,
    }


def simulate_all(
    engine: WordleEngine,
    solver,
    secrets: Optional[list[str]] = None,
    first_guess: Optional[str] = None,
    max_guesses: int = 6,
    verbose_every: int = 0,
) -> pd.DataFrame:
    """Headlessly play every word in `secrets` (defaults to the full remaining
    candidate pool) as the true secret. Returns a per-game results DataFrame;
    call .n_guesses.mean() on it for the number to compare against the
    published 3.4212 optimum."""
    secrets = secrets or engine.candidates
    records = []
    t0 = time.time()
    for i, secret in enumerate(secrets, 1):
        result = play_game(engine, 
                           solver, 
                           secret, first_guess=first_guess, max_guesses=max_guesses)
        records.append(result)
        if verbose_every and i % verbose_every == 0:
            elapsed = time.time() - t0
            print(f"  [{solver.name}] {i}/{len(secrets)} games, "
                  f"{elapsed:.1f}s elapsed, running mean="
                  f"{np.mean([r['n_guesses'] for r in records]):.3f}")
    df = pd.DataFrame(
        [{"secret": r["secret"], "n_guesses": r["n_guesses"], "solved": r["solved"]} for r in records]
    )
    return df


def summarize(df: pd.DataFrame, solver_name: str) -> dict:
    return {
        "solver": solver_name,
        "n_games": len(df),
        "mean_guesses": round(df["n_guesses"].mean(), 4),
        "median_guesses": df["n_guesses"].median(),
        "max_guesses": df["n_guesses"].max(),
        "solve_rate_within_6": round((df["solved"]).mean(), 4),
        "distribution": Counter(df["n_guesses"]),
    }
