#!/usr/bin/env python3
"""
zebra_gen.py — Infinite ZebraLogic puzzle generator focusing on the ordering
of a single target category.

**No external dependencies** — uses only Python 3 standard‑library features.

Command‑line usage
------------------
$ python zebra_gen.py -n 100 --min-houses 2 --max-houses 4 --out-prefix puzzles

Produces two files:
  • *puzzles.txt*  – printable puzzles followed by their answers
  • *puzzles.json* – list of dicts:  {"puzzle_text": …, "correct_answer": …}

Guarantees
~~~~~~~~~~
* Exactly one ordering of the chosen **target category** is deducible.
* No “item in house X” clue ever references the target category.
* Houses ∈ [min_houses, max_houses] (defaults 2‑4).
* Categories: 3–4, and never more than houses.
* Constraint types supported:  FixedHouse, SameHouse, DifferentHouse,
  NotNextTo, LeftOf, RightOf, OneHouseBetween.

Implementation strategy (vs earlier python‑constraint version)
-------------------------------------------------------------
A compact back‑tracking CSP enumerator replaces the third‑party solver.
Because puzzles are intentionally tiny (≤4 houses × ≤4 categories → 16 items),
brute‑force with early pruning is lightning‑fast and keeps the code dependency‑free.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

##########################################################
# 1.  Domain objects
##########################################################

HouseIdx = int  # 1‑based index along the street (1 … N)

@dataclass(frozen=True)
class Item:
    category: str
    name: str

    def __str__(self) -> str:  # pretty for clue text
        return self.name

##########################################################
# 2.  Constraint classes
##########################################################

class Constraint:
    """Abstract puzzle constraint."""

    def holds(self, assignment: Dict[Item, HouseIdx]) -> bool:
        """Return **True** if the constraint is *compatible* with the (possibly
        partial) assignment.  Unassigned items simply defer the check."""
        raise NotImplementedError

    def describe(self) -> str:  # human‑readable clue sentence
        raise NotImplementedError


class FixedHouse(Constraint):
    def __init__(self, item: Item, house: HouseIdx):
        self.item, self.house = item, house

    def holds(self, a):
        if self.item in a:
            return a[self.item] == self.house
        return True

    def describe(self):
        return f"The {self.item} is in house {self.house}."


class SameHouse(Constraint):
    def __init__(self, a: Item, b: Item):
        self.a, self.b = a, b

    def holds(self, a):
        return (
            self.a not in a or self.b not in a or a[self.a] == a[self.b]
        )

    def describe(self):
        return f"The {self.a} lives in the same house as the {self.b}."


class DifferentHouse(Constraint):
    def __init__(self, a: Item, b: Item):
        self.a, self.b = a, b

    def holds(self, a):
        return (
            self.a not in a or self.b not in a or a[self.a] != a[self.b]
        )

    def describe(self):
        return f"The {self.a} does not live in the same house as the {self.b}."


class NotNextTo(Constraint):
    def __init__(self, a: Item, b: Item):
        self.a, self.b = a, b

    def holds(self, a):
        return (
            self.a not in a or self.b not in a or abs(a[self.a] - a[self.b]) != 1
        )

    def describe(self):
        return f"The {self.a} is not next to the {self.b}."


class LeftOf(Constraint):
    def __init__(self, left: Item, right: Item, distance: int = 1):
        self.left, self.right, self.k = left, right, distance

    def holds(self, a):
        if self.left in a and self.right in a:
            return a[self.left] + self.k == a[self.right]
        return True

    def describe(self):
        if self.k == 1:
            return f"The {self.left} is immediately left of the {self.right}."
        return f"The {self.left} is {self.k} houses left of the {self.right}."


class RightOf(LeftOf):
    def __init__(self, right: Item, left: Item, distance: int = 1):
        super().__init__(left, right, distance)

    def describe(self):
        if self.k == 1:
            return f"The {self.right} is immediately right of the {self.left}."
        return f"The {self.right} is {self.k} houses right of the {self.left}."


class OneHouseBetween(Constraint):
    def __init__(self, a: Item, b: Item):
        self.a, self.b = a, b

    def holds(self, a):
        return (
            self.a not in a or self.b not in a or abs(a[self.a] - a[self.b]) == 2
        )

    def describe(self):
        return f"There is exactly one house between the {self.a} and the {self.b}."

##########################################################
# 3.  Constraint sampling helpers
##########################################################

_SAMPLE_TEMPLATES: List[Tuple] = []  # (callable, weight)

def _register(fn, weight: int = 1):
    _SAMPLE_TEMPLATES.append((fn, weight))
    return fn


@_register
def _sample_fixed(sol, rng, target_cat):
    item = rng.choice([i for i in sol if i.category != target_cat])
    return FixedHouse(item, sol[item])


@_register
def _sample_same(sol, rng, *_):
    return SameHouse(*rng.sample(list(sol.keys()), 2))


@_register
def _sample_diff(sol, rng, *_):
    return DifferentHouse(*rng.sample(list(sol.keys()), 2))


@_register
def _sample_not_next(sol, rng, *_):
    return NotNextTo(*rng.sample(list(sol.keys()), 2))


@_register
def _sample_leftof(sol, rng, *_):
    a, b = rng.sample(list(sol.keys()), 2)
    return LeftOf(a, b) if sol[a] < sol[b] else LeftOf(b, a)


@_register
def _sample_one_between(sol, rng, *_):
    pairs = [
        (i, j)
        for i, hi in sol.items()
        for j, hj in sol.items()
        if abs(hi - hj) == 2 and i != j
    ]
    return OneHouseBetween(*rng.choice(pairs)) if pairs else _sample_not_next(sol, rng)

##########################################################
# 4.  Item banks
##########################################################

_ITEM_BANK: Dict[str, List[str]] = {
    "Pets": ["beagle", "cat", "parakeet", "hamster", "iguana", "rabbit", "goldfish"],
    "Drinks": ["coffee", "tea", "milk", "water", "juice", "soda", "lemonade"],
    "Instruments": ["flute", "guitar", "violin", "drums", "saxophone", "trumpet", "piano"],
    "Colors": ["red", "green", "blue", "yellow", "white", "black", "orange"],
    "Desserts": ["cake", "pie", "pudding", "ice cream", "cookie", "brownie", "tart"],
}
_CATEGORIES_POOL = list(_ITEM_BANK.keys())

##########################################################
# 5.  Brute‑force enumerator (tiny CSP solver)
##########################################################

def _enumerate_target_orders(
    items: List[Item],
    houses: int,
    constraints: List[Constraint],
    target_cat: str,
    max_orders: int = 2,
):
    """Return up to *max_orders* distinct orderings for *target_cat* that satisfy
    *constraints*. Stop once *max_orders* are found (used for uniqueness test)."""
    assignment: Dict[Item, HouseIdx] = {}
    used_by_cat: Dict[str, List[HouseIdx]] = {c: [] for c in {i.category for i in items}}
    target_orders = set()

    items_sorted = sorted(items, key=lambda it: it.category)

    def backtrack(idx: int):
        if len(target_orders) >= max_orders:
            return
        if idx == len(items_sorted):
            order = tuple(
                sorted(
                    (i for i in items_sorted if i.category == target_cat),
                    key=lambda x: assignment[x],
                )
            )
            target_orders.add(order)
            return
        item = items_sorted[idx]
        cat = item.category
        for h in range(1, houses + 1):
            if h in used_by_cat[cat]:
                continue
            assignment[item] = h
            used_by_cat[cat].append(h)
            if all(c.holds(assignment) for c in constraints):
                backtrack(idx + 1)
            used_by_cat[cat].pop()
            del assignment[item]
            if len(target_orders) >= max_orders:
                return

    backtrack(0)
    return target_orders

##########################################################
# 6.  Puzzle generator
##########################################################

@dataclass
class Puzzle:
    houses: int
    categories: List[str]
    target: str
    constraints: List[Constraint]
    solution: Dict[Item, HouseIdx]

    # ----- presentation helpers -----
    def answer_string(self) -> str:
        ordered = [
            itm.name
            for itm, _ in sorted(
                ((i, h) for i, h in self.solution.items() if i.category == self.target),
                key=lambda t: t[1],
            )
        ]
        return ", ".join(ordered)

    def pretty_text(self) -> str:
        rng_local = random.Random(self.houses * 1337 + len(self.categories))
        listing = []
        for cat in self.categories:
            names = [i.name for i in self.solution if i.category == cat]
            rng_local.shuffle(names)
            listing.append(f"  {cat}: {', '.join(names)}")
        intro = (
            f"There are {self.houses} houses in a row, numbered 1 to {self.houses}.\n"
            "Each house contains exactly one item from each of the following categories,"
            " whose possible items are listed below (in no particular order):\n" +
            "\n".join(listing)
        )
        clues_txt = "\n".join(f"  • {c.describe()}" for c
