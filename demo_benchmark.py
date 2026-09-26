"""
demo_benchmark.py
================================================================================
Usage example for wordle_engine.py — copy these cells into analysis.ipynb.

"""

import time
import pandas as pd

from wordle_engine import (
    WordleEngine, load_word_pools,
    BaselineGreedyEntropySolver, MultiStepLookaheadSolver,
    InformationMetricsMatrix, play_game, simulate_all, summarize,
)

# ------------------------------------------------------------------ #
# 1. Load data (prefers ./cleaned/, falls back to the raw .txt files) #
# ------------------------------------------------------------------ #
candidates, guess_vocab = load_word_pools(".")
engine = WordleEngine(candidates, guess_vocab)
print(f"remaining candidate pool: {len(candidates)} words")
print(f"full legal guess vocabulary: {len(guess_vocab)} words")


# 2. Pick the opening word once, scanning the FULL guess vocabulary. #
#    This is the single most expensive entropy scan (~3s) — do it    #
#    once and reuse the result everywhere else instead of re-scanning#
#    12,972 words on every game in the benchmark loop.                #

t0 = time.time()
opener_finder = BaselineGreedyEntropySolver(engine, guess_pool=guess_vocab)
OPENING_WORD = opener_finder.choose(candidates)
print(f"opening word (full-vocab entropy max): '{OPENING_WORD}' "
      f"({time.time()-t0:.2f}s to compute)")


# 3. Information Metrics Matrix — 3Blue1Brown-style reference
#    for a shortlist of well-known candidate openers, PLUS the actual #
#    information received once a specific secret is revealed.         #

imm = InformationMetricsMatrix(engine)
demo_secret = candidates[0]
shortlist = sorted({OPENING_WORD, "crane", "slate", "adieu", "roate", "trace", "spare"}) # shortlist of candidate openers
metrics_table = imm.table(candidates, shortlist, secret=demo_secret)
print(f"\nInformation Metrics Matrix (secret used for 'actual' columns: '{demo_secret}')")
print(metrics_table.to_string(index=False))


# 4. Single game trace, so you can see the candidate set narrow turn  #
#    by turn (useful for the report's worked example / screenshots).  #

fast_pool_solver = BaselineGreedyEntropySolver(engine, guess_pool=candidates)
trace = play_game(engine, fast_pool_solver, secret=demo_secret, first_guess=OPENING_WORD)
print(f"\nWorked example — secret='{demo_secret}':")
for i, (g, p) in enumerate(zip(trace["guesses"], trace["patterns"]), 1):
    print(f"  guess {i}: {g}  ->  {p}")
print(f"  solved in {trace['n_guesses']} guesses" if trace["solved"] else "  NOT solved within limit")

# ------------------------------------------------------------------ #
# 5. Headless benchmark: play EVERY word in the remaining candidate   #
#    pool as the secret, for each solver. This is the number to       #
#    quote in the report and compare against the published 3.4212     #
#    optimum (computed over the larger, 2,315-word full answer list). #
#    ~25-30s each on this machine for 473 secrets.                    #
# ------------------------------------------------------------------ #
solvers = {
    "baseline": BaselineGreedyEntropySolver(engine, guess_pool=candidates),
    "lookahead": MultiStepLookaheadSolver(engine, guess_pool=candidates, top_k=8, exact_threshold=12),
}

results = []
for label, solver in solvers.items():
    t0 = time.time()
    df = simulate_all(engine, solver, first_guess=OPENING_WORD, verbose_every=0)
    elapsed = time.time() - t0
    stats = summarize(df, solver.name)
    stats["seconds"] = round(elapsed, 1)
    results.append(stats)
    print(f"\n{solver.name}: mean={stats['mean_guesses']}  "
          f"median={stats['median_guesses']}  max={stats['max_guesses']}  "
          f"({elapsed:.1f}s for {len(df)} games)")
    print(f"  distribution: {dict(sorted(stats['distribution'].items()))}")

comparison = pd.DataFrame(
    [{k: v for k, v in r.items() if k != "distribution"} for r in results]
)
print("\nSolver comparison summary:")
print(comparison.to_string(index=False))
comparison.to_csv("cleaned/solver_comparison.csv", index=False)

# ------------------------------------------------------------------ #
# Reference point for the report:
#   Selby (2022) / Bertsimas & Paskov (2024): 3.4212 average guesses,
#   exhaustive search, opening word "salet", over the FULL 2,315-word
#   answer list. Our numbers above are computed over the smaller
#   473-word REMAINING pool (already-used NYT answers excluded), which
#   is why they land below 3.42 — it's a genuinely easier instance of
#   the same problem, not an apples-to-apples beat of the optimum.
# ------------------------------------------------------------------ #
