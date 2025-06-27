#!/usr/bin/env python3
"""Bridge‑and‑Torch ‘Gossip’ puzzle generator.

For each instance the script produces a natural‑language question that
resembles the classic bridge‑and‑torch problem, except that some players
*exaggerate* their published speeds.  The solver must find the minimum
actual total crossing time.  Each generated puzzle is guaranteed to have
*exactly one* minimal achievable time under the stated rules.

Output: a single JSON file containing a list of objects of the form
    {"question": <str>, "canonical_answer": <str>}

Example CLI::

    python bridge_torch_generator.py --count 10 --seed 42 --output puzzles/

The file puzzles/bridge_torch_gossip_10_42.json will be created (folders
are made on demand).
"""
from __future__ import annotations

import argparse
import json
import random
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def bridge_time(times: List[int]) -> int:
    """Return the minimal total crossing time for *times* (classic algorithm).

    Parameters
    ----------
    times : list[int]
        Positive integers – individual true crossing times.

    Returns
    -------
    int
        Minimal total time for everyone to cross.
    """
    times = sorted(times)
    total = 0
    n = len(times)

    while n > 3:
        t1, t2 = times[0], times[1]
        tn_1, tn = times[-2], times[-1]
        # Strategy A: fastest two shuttle
        option_a = 2 * t2 + t1 + tn
        # Strategy B: fastest shuttles individually
        option_b = 2 * t1 + tn_1 + tn
        total += min(option_a, option_b)
        # Two slowest are now across; remove them
        times.pop()  # remove largest
        times.pop()  # remove second‑largest (now last)
        n -= 2
    # Handle final <=3 people
    if n == 3:
        total += sum(times)
    elif n == 2:
        total += times[1]
    elif n == 1:
        total += times[0]
    return total


def english_join(nums: List[int]) -> str:
    """Return comma‑separated English list: "5, 8, 10 and 13"."""
    if not nums:
        return ""
    if len(nums) == 1:
        return str(nums[0])
    return ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"


# ---------------------------------------------------------------------------
# Puzzle generator core
# ---------------------------------------------------------------------------

def _candidate_instance(rng: random.Random) -> Tuple[str, str]:
    """Generate one puzzle (question, answer) – may *not* be unique yet."""
    # 1. Basic parameters
    n = rng.randint(2, 4)  # number of people
    exaggeration = rng.randint(1, 5)  # +d minutes for exaggerators
    # generate strictly increasing *true* times (avoid duplicates for clarity)
    base_times = sorted(rng.sample(range(1, 20), n))

    # choose number of exaggerators (at least 1, not all)
    k = rng.randint(1, n - 1)

    # pick which indices exaggerate for *this* hidden ground truth
    ground_truth_subset = set(rng.sample(range(n), k))

    # claimed times list (parallel to base_times)
    claimed = [
        t + exaggeration if i in ground_truth_subset else t for i, t in enumerate(base_times)
    ]

    # 2. Compute *minimal* total time achievable **given constraints only**
    #    by enumerating all subsets of size k (unknown to solver).
    min_times = []  # list of (total_time, subset) for minimal candidates
    min_value = None
    for subset in combinations(range(n), k):
        real_times = [
            claimed[i] - exaggeration if i in subset else claimed[i] for i in range(n)
        ]
        total = bridge_time(real_times)
        if (min_value is None) or (total < min_value):
            min_value = total
            min_times = [(total, subset)]
        elif total == min_value:
            min_times.append((total, subset))

    # 3. Accept instance only if the minimal achievable *value* is unique.
    if len({t for t, _ in min_times}) != 1:
        raise ValueError("minimal time not unique – regenerate")

    question = (
        f"There are {n} people who must cross a narrow bridge at night. "
        "They have one torch, and at most two can cross at a time. The torch "
        "must be carried whenever someone crosses, and when two cross together "
        "they move at the pace of the slower person.\n\n"
        f"Their *claimed* individual crossing times (in minutes) are: {english_join(sorted(claimed))}. "
        f"Exactly {k} of them are exaggerating by **adding {exaggeration} minute{'s' if exaggeration>1 else ''}** to their real time. "
        "Everyone else is telling the truth.\n\n"
        "What is the *minimum* total time, in minutes, that this crossing could actually take?"
    )

    canonical_answer = str(min_value)
    return question, canonical_answer


def generate_puzzles(count: int, seed: Optional[int] = None) -> List[dict]:
    """Return a list of *count* valid puzzle dicts."""
    rng = random.Random(seed)
    puzzles: List[dict] = []
    attempts = 0
    while len(puzzles) < count:
        attempts += 1
        try:
            q, a = _candidate_instance(rng)
            puzzles.append({"question": q, "canonical_answer": a})
        except ValueError:
            # uniqueness failed – try again
            continue
        # fail‑safe: avoid infinite loop if constraints impossible
        if attempts > count * 1000:
            raise RuntimeError("Too many attempts – adjust parameters")
    return puzzles


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point parsed to match user specification."""
    ap = argparse.ArgumentParser(description="Generate bridge‑torch‑gossip puzzles.")
    ap.add_argument("--count", type=int, default=100, help="How many puzzles to generate")
    ap.add_argument("--seed", type=int, default=None, help="Random‑seed for reproducibility")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Directory to write the resulting JSON file",
    )
    args = ap.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    filename = args.output / (
        f"bridge_torch_gossip_{args.count}_{args.seed if args.seed is not None else 'rand'}.json"
    )

    puzzles = generate_puzzles(count=args.count, seed=args.seed)
    with filename.open("w", encoding="utf‑8") as fp:
        json.dump(puzzles, fp, indent=2, ensure_ascii=False)

    print(f"✔ Wrote {len(puzzles)} puzzles to {filename}")


if __name__ == "__main__":  # pragma: no cover
    main()
