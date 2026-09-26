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


def load_full_answer_list(root: str | Path = ".") -> list[str]:
    """The full published answer list (2,315 words), WITHOUT subtracting
    already-used past answers. Use this — not load_word_pools()'s reduced
    pool — whenever you want a number that's directly comparable to
    published results (Selby 2022 / Bertsimas & Paskov 2024: 3.4212 average
    guesses over this exact list)."""
    paths = resolve_paths(root)
    return _read_words(paths["raw_answers"])


def load_frequency_weights(words: list[str], lang: str = "en") -> dict[str, float]:
    """Real-world word-frequency prior, via the `wordfreq` package (Robyn
    Speer et al.) — built from a mix of Wikipedia, subtitles, news, Twitter,
    and book text (the "Exquisite Corpus"), not a synthetic or invented
    weighting. `wordfreq.zipf_frequency` returns log10(occurrences per
    billion tokens) + 3; we invert that back to a linear frequency estimate
    (10**zipf) to use as an (unnormalized) Bayesian prior mass — words that
    are more common in real English text get proportionally more prior
    probability of being the day's answer, which is a genuine modeling
    choice about how NYT editors actually pick words, not just a synthetic
    reweighting invented for this project.

    Raises ImportError with a clear message if wordfreq isn't installed
    (`pip install wordfreq`).
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError as e:
        raise ImportError(
            "load_frequency_weights() needs the `wordfreq` package: "
            "run `pip install wordfreq` (or `pip install wordfreq "
            "--break-system-packages` in a managed environment)."
        ) from e
    weights = {}
    for w in words:
        z = zipf_frequency(w, lang)
        weights[w] = 10 ** z if z > 0 else 10 ** 0.5  # tiny floor for zero-frequency words
    return weights



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

    def weighted_entropy(self, guess: str, candidate_words: list[str], weights: dict[str, float]) -> float:
        """Bayesian version of entropy(): buckets are weighted by prior mass
        (e.g. real word frequency) instead of raw counts, so the expected
        information gain reflects how likely each resulting bucket actually
        is under the prior, not just how many words happen to fall in it."""
        buckets = self.partition(guess, candidate_words)
        total = sum(weights.get(w, 0.0) for w in candidate_words)
        if total <= 0:
            return 0.0
        ent = 0.0
        for b in buckets.values():
            wsum = sum(weights.get(w, 0.0) for w in b)
            if wsum <= 0:
                continue
            p = wsum / total
            ent -= p * math.log2(p)
        return ent

    # ------------------------------------------------------------------
    # Vectorized fast path: precompute the full (guess x candidate) pattern
    # matrix ONCE, then score every guess in the pool in a single numpy call
    # per turn instead of a Python loop with per-guess dict-building. This is
    # what makes benchmarking across the full 2,315-word answer list
    # tractable (a plain Python loop over guess_pool for every turn of every
    # game does not finish in reasonable time at that scale).
    # ------------------------------------------------------------------

    def build_fast_index(self, guess_pool: Optional[list[str]] = None) -> None:
        """Precompute the (len(guess_pool) x len(self.candidates)) pattern
        matrix. Must be called once before best_guess_fast(). `guess_pool`
        defaults to self.candidates (the common "candidates-only" fast mode

        used per-turn after an initial full-vocab opener)."""
        guess_pool = guess_pool or self.candidates
        self._fast_guess_pool = list(guess_pool)
        self._fast_col_index = {w: i for i, w in enumerate(self.candidates)}
        g_arr = words_to_array(self._fast_guess_pool)
        G = g_arr.shape[0]
        M = np.empty((G, self._cand_arr.shape[0]), dtype=np.int16)
        for i in range(G):
            M[i] = feedback_batch(g_arr[i], self._cand_arr)
        self._fast_matrix = M

    def best_guess_fast(
        self, candidate_subset: list[str], weights: Optional[dict[str, float]] = None
    ) -> tuple[str, float]:
        """Vectorized equivalent of scanning `self._fast_guess_pool` with
        entropy()/weighted_entropy(), using a single bincount-based numpy
        pass instead of a per-guess Python loop. `candidate_subset` MUST be a
        subset of `self.candidates` (true by construction during any game,
        since filtering only ever removes words)."""
        if not hasattr(self, "_fast_matrix"):
            raise RuntimeError("call engine.build_fast_index() before best_guess_fast().")
        cols = [self._fast_col_index[w] for w in candidate_subset]
        sub = self._fast_matrix[:, cols].astype(np.int64)  # (G, k)
        G, k = sub.shape

        if weights is None:
            w_sub, total = None, float(k)
        else:
            w_sub = np.array([weights.get(w, 0.0) for w in candidate_subset], dtype=np.float64)
            total = float(w_sub.sum())

        offsets = (np.arange(G) * N_PATTERNS)[:, None]
        flat_idx = (sub + offsets).ravel()
        if w_sub is None:
            counts_flat = np.bincount(flat_idx, minlength=G * N_PATTERNS)
        else:
            counts_flat = np.bincount(flat_idx, weights=np.tile(w_sub, G), minlength=G * N_PATTERNS)
        counts = counts_flat.reshape(G, N_PATTERNS)

        if total <= 0:
            return self._fast_guess_pool[0], 0.0
        probs = counts / total
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
        ent = -terms.sum(axis=1)
        # Tie-break like the slow path does: among guesses tied for max entropy,
        # prefer one that's an actual remaining candidate — it has the same
        # expected information but a nonzero chance of winning outright.
        # (Bug fix: a plain argmax here silently favored whichever guess came
        # first in guess_pool order — e.g. "abled", first alphabetically —
        # any time it tied for max entropy, which was often. That wasted a
        # real guess in ~16% of games in testing and inflated every full-list
        # average reported before this fix.)
        best_val = float(ent.max())
        tied = np.where(ent >= best_val - 1e-9)[0]
        cand_set = set(candidate_subset)
        preferred = [i for i in tied if self._fast_guess_pool[i] in cand_set]
        best_row = preferred[0] if preferred else int(tied[0])
        return self._fast_guess_pool[best_row], float(ent[best_row])


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


class BayesianFrequencySolver:
    """Bayesian layer (Lecture 5) on top of the same informed-search
    machinery. The candidate set is still narrowed by Bayes' rule the usual
    way for Wordle: since feedback is deterministic given the secret,
    P(feedback | w) is 1 for words consistent with it and 0 otherwise, so the
    posterior is just the prior restricted to consistent words and
    renormalized — exactly the `[w for w in candidates if feedback(guess,w)
    == pattern]` filtering step already in play_game(). What THIS class adds
    is using a non-uniform PRIOR (real word-frequency weights) instead of
    treating every remaining candidate as equally likely, in two ways:

    1. Weighted entropy: the guess-selection heuristic maximizes expected
       information under the weighted posterior, not a uniform one.
    2. MAP short-circuit: if the single most probable remaining candidate
       already holds more than `map_guess_threshold` of the posterior mass,
       guess it directly instead of continuing to optimize for information —
       once one word is very likely correct, spending the turn trying to
       "learn more" has lower expected value than just trying it.
    """

    name = "Bayesian Frequency-Weighted"

    def __init__(
        self,
        engine: WordleEngine,
        weights: dict[str, float],
        guess_pool: Optional[list[str]] = None,
        map_guess_threshold: float = 0.30,
    ):
        self.engine = engine
        self.weights = weights
        self.guess_pool = guess_pool or engine.guess_vocab
        self.map_guess_threshold = map_guess_threshold

    def posterior(self, candidates: list[str]) -> dict[str, float]:
        total = sum(self.weights.get(w, 0.0) for w in candidates)
        if total <= 0:
            return {w: 1.0 / len(candidates) for w in candidates}
        return {w: self.weights.get(w, 0.0) / total for w in candidates}

    def choose(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError(
                "No candidates remain consistent with the feedback so far. This means "
                "either the secret isn't in the engine's candidate pool or there's a "
                "feedback-consistency bug upstream."
            )
        if len(candidates) == 1:
            return candidates[0]

        post = self.posterior(candidates)
        map_word, map_prob = max(post.items(), key=lambda kv: kv[1])
        if map_prob >= self.map_guess_threshold:
            return map_word

        best_word, best_score = None, -1.0
        for g in self.guess_pool:
            score = self.engine.weighted_entropy(g, candidates, self.weights)
            if score > best_score or (score == best_score and g in candidates and best_word not in candidates):
                best_word, best_score = g, score
        return best_word


class EndgameExactSolver:
    """Exact expectiminimax search, but ONLY once the candidate set is small
    (<= `threshold`) — this is the "endgame lookahead" piece of the plan: it
    wraps any other solver as a `delegate` and takes over just for the final
    stretch of the game, where brute force is cheap and exact.

    Recurrence (E(S) = expected additional guesses to fully identify the
    secret from candidate set S):
        E({w}) = 1                      (just guess it)
        E(S)   = 1 + min_g sum_p (|S_p|/|S|) * cost(S_p)
                 where S_p is the bucket feedback pattern p produces, and
                 cost(S_p) = 0 if S_p == {g} (g itself matched -> already
                 solved by this guess, no recursion needed) else E(S_p).
    Results are memoized on the exact candidate SET, so the same sub-problem
    encountered from different games (or different branches of the same
    game) is only ever solved once for the lifetime of this object.

    "Adaptable for further tuning": nothing about this class is tied to a
    specific heuristic — swap `delegate` for BaselineGreedyEntropySolver,
    MultiStepLookaheadSolver, or BayesianFrequencySolver and only the
    large-candidate-set behavior changes; the exact endgame logic is
    reused unmodified. `threshold` and `guess_options` are also both
    free parameters to sweep when tuning.
    """

    name = "Endgame-Exact"

    def __init__(
        self,
        engine: WordleEngine,
        delegate,
        threshold: int = 8,
        guess_options: Optional[list[str]] = None,
    ):
        self.engine = engine
        self.delegate = delegate
        self.threshold = threshold
        self.guess_options = guess_options  # None => use the candidate set itself each call
        self._memo: dict[frozenset, tuple[str, float]] = {}

    def choose(self, candidates: list[str]) -> str:
        if not candidates:
            raise ValueError(
                "No candidates remain consistent with the feedback so far. This means "
                "either the secret isn't in the engine's candidate pool or there's a "
                "feedback-consistency bug upstream."
            )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) <= self.threshold:
            guess_options = self.guess_options or candidates
            best_guess, _ = self._exact(frozenset(candidates), guess_options)
            return best_guess
        return self.delegate.choose(candidates)

    def _exact(self, candidate_set: frozenset, guess_options: list[str]) -> tuple[str, float]:
        cached = self._memo.get(candidate_set)
        if cached is not None:
            return cached

        candidates = list(candidate_set)
        n = len(candidates)
        if n == 1:
            result = (candidates[0], 1.0)
            self._memo[candidate_set] = result
            return result

        best_guess, best_cost = None, math.inf
        for g in guess_options:
            buckets = self.engine.partition(g, candidates)
            if len(buckets) == 1:
                continue  # zero information against this particular subset — never worth recursing into
            cost = 1.0
            for bucket in buckets.values():
                if len(bucket) == 1 and bucket[0] == g:
                    continue  # this outcome means g WAS the secret — no extra guesses needed
                p = len(bucket) / n
                _, sub_cost = self._exact(frozenset(bucket), guess_options)
                cost += p * sub_cost
            if cost < best_cost:
                best_guess, best_cost = g, cost

        result = (best_guess, best_cost)
        self._memo[candidate_set] = result
        return result


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
        result = play_game(engine, solver, secret, first_guess=first_guess, max_guesses=max_guesses)
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


# ==============================================================================
# 7. FAST-PATH (vectorized) game loop + benchmark — for full 2,315-word runs
# ==============================================================================

def play_game_fast(
    engine: WordleEngine,
    secret: str,
    weights: Optional[dict[str, float]] = None,
    first_guess: Optional[str] = None,
    max_guesses: int = 6,
    require_valid_secret: bool = True,
) -> dict:
    """Same contract as play_game(), but uses engine.best_guess_fast()
    (requires engine.build_fast_index() to have been called first) instead of
    a Solver object. `weights=None` reproduces plain greedy-entropy;
    `weights={...}` reproduces the Bayesian frequency-weighted heuristic —
    both via the identical vectorized code path, so timing/behavior
    differences between the two are purely about the weighting, not about
    two different implementations."""
    if require_valid_secret and secret not in engine.candidates:
        raise ValueError(f"'{secret}' is not in the engine's candidate pool.")
    candidates = list(engine.candidates)
    guesses_made: list[str] = []
    patterns_seen: list[str] = []

    for turn in range(1, max_guesses + 1):
        if turn == 1 and first_guess:
            guess = first_guess
        elif len(candidates) == 1:
            guess = candidates[0]
        else:
            guess, _ = engine.best_guess_fast(candidates, weights=weights)
        pattern = engine.feedback(guess, secret)
        guesses_made.append(guess)
        patterns_seen.append(pattern_to_str(pattern))

        if guess == secret:
            return {"secret": secret, "guesses": guesses_made, "patterns": patterns_seen,
                    "n_guesses": turn, "solved": True}
        candidates = [w for w in candidates if engine.feedback(guess, w) == pattern]

    return {"secret": secret, "guesses": guesses_made, "patterns": patterns_seen,
            "n_guesses": max_guesses, "solved": False}


def simulate_all_fast(
    engine: WordleEngine,
    weights: Optional[dict[str, float]] = None,
    secrets: Optional[list[str]] = None,
    first_guess: Optional[str] = None,
    max_guesses: int = 6,
    verbose_every: int = 0,
    solver_name: str = "fast",
) -> pd.DataFrame:
    """Fast-path equivalent of simulate_all(): plays every word in `secrets`
    (default: engine.candidates) as the secret, returning one row per game
    INCLUDING the full guess sequence, so results can be joined side-by-side
    across solvers by `secret`."""
    secrets = secrets or engine.candidates
    records = []
    t0 = time.time()
    for i, secret in enumerate(secrets, 1):
        r = play_game_fast(engine, secret, weights=weights, first_guess=first_guess, max_guesses=max_guesses)
        records.append(r)
        if verbose_every and i % verbose_every == 0:
            elapsed = time.time() - t0
            running_mean = np.mean([x["n_guesses"] for x in records])
            print(f"  [{solver_name}] {i}/{len(secrets)} games, {elapsed:.1f}s elapsed, "
                  f"running mean={running_mean:.3f}")
    return pd.DataFrame(
        [{"secret": r["secret"], "n_guesses": r["n_guesses"], "solved": r["solved"],
          "guesses": r["guesses"]} for r in records]
    )


def side_by_side(*named_dfs: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """Join two or more simulate_all_fast() result frames on `secret` into one
    comparison table: n_guesses and the actual guess sequence for each named
    solver, plus the difference in guess count between the first two."""
    merged = None
    for name, df in named_dfs:
        renamed = df.rename(columns={
            "n_guesses": f"{name}_n_guesses",
            "guesses": f"{name}_guesses",
            "solved": f"{name}_solved",
        })
        merged = renamed if merged is None else merged.merge(renamed, on="secret")
    names = [n for n, _ in named_dfs]
    if len(names) >= 2:
        merged["diff"] = merged[f"{names[0]}_n_guesses"] - merged[f"{names[1]}_n_guesses"]
    return merged


def find_best_opener(engine: WordleEngine) -> tuple[str, float]:
    """The "computed opener": whichever word in engine.candidates maximizes
    1-ply entropy against the full candidate pool. Requires
    engine.build_fast_index(guess_pool=engine.candidates) to have been called
    first (cheap — see demo_benchmark.py)."""
    return engine.best_guess_fast(engine.candidates)


def compare_openers(
    engine: WordleEngine,
    opener_a: str,
    opener_b: str,
    secrets: Optional[list[str]] = None,
    weights: Optional[dict[str, float]] = None,
    max_guesses: int = 6,
) -> dict:
    """Head-to-head: play every word in `secrets` (default: the full
    candidate pool) as the secret with each opener, using identical solver
    logic apart from the forced first guess. Returns per-opener summaries
    plus a word-by-word side-by-side table — this is the direct way to check
    "is my chosen opener actually competitive with the computed one, or by
    how much does it lag" rather than trusting entropy bits alone (a higher
    first-guess entropy doesn't always translate 1:1 into fewer average
    guesses, since it says nothing about how informative the FOLLOW-UP
    guesses end up being)."""
    secrets = secrets or engine.candidates
    df_a = simulate_all_fast(engine, weights=weights, secrets=secrets, first_guess=opener_a, max_guesses=max_guesses)
    df_b = simulate_all_fast(engine, weights=weights, secrets=secrets, first_guess=opener_b, max_guesses=max_guesses)
    stats_a = summarize(df_a, opener_a)
    stats_b = summarize(df_b, opener_b)
    comparison = side_by_side((opener_a, df_a), (opener_b, df_b))
    return {
        "opener_a": stats_a,
        "opener_b": stats_b,
        "opener_a_better_on": int((comparison["diff"] < 0).sum()),  # a needed FEWER guesses
        "opener_b_better_on": int((comparison["diff"] > 0).sum()),
        "tied_on": int((comparison["diff"] == 0).sum()),
        "comparison_table": comparison,
    }