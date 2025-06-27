#!/usr/bin/env python3
"""Generate text‑only *Logic‑Locker PIN* puzzles.

Each puzzle is a natural‑language set of clues that pin down **exactly one**
numeric code.  Output is a JSON list with objects of the form
    { "question": <str>, "canonical_answer": <str> }

Changes from the first version
------------------------------
* **Robust uniqueness** – Factories can be used repeatedly; we keep trying
  new clues until uniqueness *or* we run out of room, dramatically reducing
  "failed to generate" cases.
* **Extra clue type** – Arbitrary positional digit (“the third digit is 7”)
  provides more variety and greater pruning power.
* **Tidier wording** – Small language tweaks for smoother reading.
* **Config knobs** – `--max-attempts` and `--max-clues` are now CLI flags.

Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------
# Helper types and utilities
# ---------------------------------------------------------

Predicate = Callable[[str], bool]      # code → True/False
Clue = Tuple[str, Predicate]           # (text, predicate)


ORDINAL = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth",
]


def all_codes(length: int) -> List[str]:
    """Return every _length_-digit string; no leading zeros allowed."""
    return [str(first) + "".join(map(str, tail))
            for first in range(1, 10)
            for tail in product(range(10), repeat=length - 1)]


def digits(code: str) -> List[int]:
    return [int(ch) for ch in code]

# ---------------------------------------------------------
# Clue factories – each produces (text, predicate) for a given secret.
# ---------------------------------------------------------



def clue_even_count(secret: str) -> Clue:
    k = sum(int(ch) % 2 == 0 for ch in secret)
    txt = f"Exactly {k} digit{'s' if k != 1 else ''} {'is' if k == 1 else 'are'} even."
    return txt, lambda code, k=k: sum(int(c) % 2 == 0 for c in code) == k


def first_two_sum(secret: str) -> Clue:
    s = int(secret[0]) + int(secret[1])
    txt = f"The first two digits add up to {s}."
    return txt, lambda code, s=s: int(code[0]) + int(code[1]) == s

def last_two_sum(secret: str) -> Clue:
    s = int(secret[-2]) + int(secret[-1])
    txt = f"The last two digits add up to {s}."
    return txt, lambda code, s=s: int(code[-2]) + int(code[-1]) == s

def first_and_third_sum(secret: str) -> Clue:
    s = int(secret[0]) + int(secret[2])
    txt = f"The first and third digits add up to {s}."
    return txt, lambda code, s=s: int(code[0]) + int(code[2]) == s

def first_and_last_sum(secret: str) -> Clue:
    s = int(secret[0]) + int(secret[-1])
    txt = f"The first and last digits add up to {s}."
    return txt, lambda code, s=s: int(code[0]) + int(code[-1]) == s


def clue_sum(secret: str) -> Clue:
    s = sum(digits(secret))
    txt = f"The digits add up to {s}."
    return txt, lambda code, s=s: sum(int(c) for c in code) == s


def clue_distinct_count(secret: str) -> Clue:
    k = len(set(secret))
    txt = f"Exactly {k} distinct digit{'s' if k != 1 else ''} appear."
    return txt, lambda code, k=k: len(set(code)) == k


def clue_position_digit(secret: str) -> Clue:
    pos = random.randrange(len(secret))  # 0‑based
    d = secret[pos]
    ord_word = ORDINAL[pos] if pos < len(ORDINAL) else f"{pos+1}th"
    txt = f"The {ord_word} position digit is {d}."
    return txt, lambda code, pos=pos, d=d: code[pos] == d

def clue_first_last_relation(secret: str) -> Clue:
    first, last = int(secret[0]), int(secret[-1])
    diff = first - last
    if diff == 0:
        txt = "The first and last digits are equal."
        return txt, lambda code: code[0] == code[-1]
    direction = "greater" if diff > 0 else "smaller"
    txt = f"The first digit is {abs(diff)} {direction} than the last digit."
    return txt, lambda code, diff=diff: (int(code[0]) - int(code[-1])) == diff

def clue_digit_range(secret: str) -> Clue:
    min_d, max_d = min(digits(secret)), max(digits(secret))
    txt = f"The smallest digit is {min_d} and the largest is {max_d}."
    return txt, lambda code, min_d=min_d, max_d=max_d: min(int(c) for c in code) == min_d and max(int(c) for c in code) == max_d


CLUE_FACTORIES: List[Callable[[str], Clue]] = [
    clue_even_count,
    clue_sum,
    first_two_sum,
    last_two_sum,
    first_and_third_sum,
    first_and_last_sum,
    clue_distinct_count,
    clue_digit_range,
    #clue_position_digit,
    clue_first_last_relation,
]

# ---------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------


def generate_single(*, length: int, universe: List[str], max_clues: int = 30,
                    hard_retry: int = 100) -> Tuple[str, str]:
    """Return (question, answer).  May raise RuntimeError after many retries."""
    for _ in range(hard_retry):
        secret = random.choice(universe)
        candidates = set(universe)
        chosen_clues: List[str] = []
        predicates: List[Predicate] = []
        seen_texts: set[str] = set()


        while len(candidates) > 1 and len(chosen_clues) < max_clues:
            factory = random.choice(CLUE_FACTORIES)
            clue_txt, pred = factory(secret)

            if clue_txt in seen_texts:
                continue  # duplicate wording, skip
            new_candidates = {c for c in candidates if pred(c)}
            if secret not in new_candidates or len(new_candidates) == len(candidates):
                continue  # broken or unhelpful clue
            # Accept
            chosen_clues.append(clue_txt)
            predicates.append(pred)
            seen_texts.add(clue_txt)
            candidates = new_candidates

        if len(candidates) == 1:
            bullet_list = "\n".join(f"  • {t}" for t in chosen_clues)
            question = (
                f"Find the {length}-digit code that unlocks the locker.\n"
                f"It satisfies all of the following conditions:\n{bullet_list}"
                "You can think step by step as long as you'd like."
                "Give your final answer inside a \\boxed{...} tag."
            )
            return question, secret
    raise RuntimeError("Unable to build a unique puzzle after many attempts – try relaxing settings.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate logic locker puzzles.")
    ap.add_argument("--min-length", type=int, default=5, help="Number length")
    ap.add_argument("--max-length", type=int, default=6, help="Number length")
    ap.add_argument("--count", type=int, default=100, help="How many puzzles to generate")
    ap.add_argument("--seed", type=int, default=None, help="Random‑seed for reproducibility")
    ap.add_argument("--output", type=Path, default=Path("."), help="Directory for output files")
    ap.add_argument("--max-clues", type=int, default=20, help="Upper bound on clues per puzzle")
    ap.add_argument("--max-attempts", type=int, default=100, help="Retries per puzzle if unlucky")

    args = ap.parse_args(argv)
    if args.min_length < 2:
        ap.error("--min-length must be at least 2")
    if args.max_length < args.min_length:
        ap.error("--max-length must be at least --min-length")
    if args.count < 1:
        ap.error("--count must be positive")

    if args.seed is not None:
        random.seed(args.seed)

    puzzles: List[dict[str, str]] = []

    for i in range(args.count):
        print(f"Generating puzzle {i + 1}/{args.count}")
        try:
            length = random.randint(args.min_length, args.max_length)
            universe = all_codes(length)
            q, a = generate_single(length=length, universe=universe,
                                   max_clues=args.max_clues, hard_retry=args.max_attempts)
            puzzles.append({"question": q, "canonical_answer": a})
        except RuntimeError as er:
            print(f"[warn] Skipping puzzle {i}: {er}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    outfile = args.output / "logic_lockers.json"
    with outfile.open("w", encoding="utf‑8") as fh:
        json.dump(puzzles, fh, indent=2)

    print(f"Wrote {len(puzzles)} puzzle(s) to {outfile}")


if __name__ == "__main__":
    main()
