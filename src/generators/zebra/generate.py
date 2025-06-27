# zebra_generator.py
"""ZebraLogic Puzzle Generator – single‑file **pure‑Python** implementation
================================================================================
Generates an **infinite stream** of Zebra/Einstein‑style logic puzzles whose goal
is to deduce the ordering of **one chosen category**.  The program *only* uses
Python ≥3.10 standard‑library modules.

Key features
------------
* **Variable size** – puzzle‑wide house‑count picked per puzzle from a user‑
  supplied range or fixed value (CLI flags).
* **Rich clue palette** – implements 10 rule types:
  * `IsAt`, `NotAt`
  * `SameHouse`, `DiffHouse`
  * `Adjacent`, `NotAdjacent`
  * `LeftOf`, `RightOf`
  * `ExactlyKBetween` "exactly *k* houses between A and B" (k≥1)
  * `OneOfSetAt` XOR‑style: "exactly **one** of (A,B,C) is in house h".
* **Minimal clue set** – constraints are added until *and only until* the target
  category becomes uniquely determined; then a greedy pass removes redundancies.
* **Dual export** – every puzzle is written to **JSON** (always with answer key)
  *and* a human‑readable plain‑text file.
* **Extensible** – add new rule types by sub‑classing `Constraint`; nothing else
  in the engine needs to change.

CLI synopsis
------------
```bash
python zebra_generator.py \
    --count 5 \
    --houses-min 4 --houses-max 6 \
    --output puzzles \
    --seed 42
```
See `python zebra_generator.py --help` for the full option list.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1 – Domain model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    name: str
    category: str

    def __str__(self) -> str:  # pragma: no cover
        return self.name


@dataclass(frozen=True)
class House:
    index: int  # 1‑based

    def __str__(self) -> str:  # pragma: no cover
        return f"House {self.index}"


@dataclass
class Category:
    name: str
    items: List[Item]  # exactly |houses| items at generation time

    def __iter__(self):  # type: ignore[override]
        return iter(self.items)


# ---------------------------------------------------------------------------
# 2 – Constraint hierarchy (each rule type = one subclass)
# ---------------------------------------------------------------------------


class Constraint:
    """Abstract base class for every clue/rule."""

    rule_name: str = "<abstract>"

    # ---- public interface -------------------------------------------------
    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        """Return **False** the moment *assignment* violates the rule.

        `assignment` may be *partial* during back‑tracking: some houses may lack
        some categories.  Implementations must therefore allow "unknown" values.
        """
        raise NotImplementedError

    def to_text(self) -> str:
        """Plain‑English rendering for the text export."""
        raise NotImplementedError

    def to_json(self) -> Dict:
        """Minimal JSON representation (type + operands)."""
        raise NotImplementedError

    # ---- helpers for subclasses ------------------------------------------
    @staticmethod
    def _item_location(item_name: str, assignment: Dict[int, Dict[str, Item]]) -> Optional[int]:
        """Return the **house index** where *item_name* currently sits, or *None* if
        the item isn't yet placed in the partial assignment."""
        for h_idx, cat_map in assignment.items():
            for itm in cat_map.values():
                if itm.name == item_name:
                    return h_idx
        return None

    # nice repr for debugging ------------------------------------------------
    def __repr__(self):  # pragma: no cover
        return f"<{self.rule_name}: {self.to_text()}>"


# ----- concrete rule subclasses -------------------------------------------


def _plural(n: int, singular: str, plural: str) -> str:  # small helper
    return singular if n == 1 else plural


@dataclass
class IsAt(Constraint):
    item: str
    house_idx: int
    rule_name: str = "IsAt"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        loc = self._item_location(self.item, assignment)
        return loc is None or loc == self.house_idx

    def to_text(self) -> str:
        return f"The {self.item} is in house {self.house_idx}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item": self.item, "house": self.house_idx}


@dataclass
class NotAt(Constraint):
    item: str
    house_idx: int
    rule_name: str = "NotAt"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        loc = self._item_location(self.item, assignment)
        return loc is None or loc != self.house_idx

    def to_text(self) -> str:
        return f"The {self.item} is not in house {self.house_idx}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item": self.item, "house": self.house_idx}


@dataclass
class SameHouse(Constraint):
    item_a: str
    item_b: str
    rule_name: str = "SameHouse"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or a == b

    def to_text(self) -> str:
        return f"The {self.item_a} is in the same house as the {self.item_b}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item_a": self.item_a, "item_b": self.item_b}


@dataclass
class DiffHouse(Constraint):
    item_a: str
    item_b: str
    rule_name: str = "DiffHouse"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or a != b

    def to_text(self) -> str:
        return f"The {self.item_a} is not in the same house as the {self.item_b}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item_a": self.item_a, "item_b": self.item_b}


@dataclass
class Adjacent(Constraint):
    item_a: str
    item_b: str
    rule_name: str = "Adjacent"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or abs(a - b) == 1

    def to_text(self) -> str:
        return f"The {self.item_a} is immediately next to the {self.item_b}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item_a": self.item_a, "item_b": self.item_b}


@dataclass
class NotAdjacent(Constraint):
    item_a: str
    item_b: str
    rule_name: str = "NotAdjacent"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or abs(a - b) != 1

    def to_text(self) -> str:
        return f"The {self.item_a} is not next to the {self.item_b}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item_a": self.item_a, "item_b": self.item_b}


@dataclass
class LeftOf(Constraint):
    item_a: str
    item_b: str
    rule_name: str = "LeftOf"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or a < b

    def to_text(self) -> str:
        return f"The {self.item_a} is somewhere to the left of the {self.item_b}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item_a": self.item_a, "item_b": self.item_b}


@dataclass
class RightOf(Constraint):
    item_a: str
    item_b: str
    rule_name: str = "RightOf"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or a > b

    def to_text(self) -> str:
        return f"The {self.item_a} is somewhere to the right of the {self.item_b}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "item_a": self.item_a, "item_b": self.item_b}


@dataclass
class ExactlyKBetween(Constraint):
    item_a: str
    item_b: str
    k: int  # ≥1
    rule_name: str = "ExactlyKBetween"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        a = self._item_location(self.item_a, assignment)
        b = self._item_location(self.item_b, assignment)
        return a is None or b is None or abs(a - b) - 1 == self.k

    def to_text(self) -> str:
        return (
            f"There are exactly {self.k} {_plural(self.k, 'house', 'houses')} "
            f"between the {self.item_a} and the {self.item_b}."
        )

    def to_json(self) -> Dict:
        return {
            "type": self.rule_name,
            "item_a": self.item_a,
            "item_b": self.item_b,
            "k": self.k,
        }


# XOR‑style rule -----------------------------------------------------------


@dataclass
class OneOfSetAt(Constraint):
    items: Tuple[str, str, str]  # exactly 3 distinct items
    house_idx: int
    rule_name: str = "OneOfSetAt"

    def is_satisfied(self, assignment: Dict[int, Dict[str, Item]]) -> bool:
        count_at = 0
        unknown = 0
        for itm in self.items:
            loc = self._item_location(itm, assignment)
            if loc is None:
                unknown += 1
            elif loc == self.house_idx:
                count_at += 1
        # if more than one already placed here → violation
        if count_at > 1:
            return False
        # when all three placed, exactly one must be here
        if unknown == 0 and count_at != 1:
            return False
        return True

    def to_text(self) -> str:
        a, b, c = self.items
        return f"Exactly one of {a}, {b}, or {c} is in house {self.house_idx}."

    def to_json(self) -> Dict:
        return {"type": self.rule_name, "items": list(self.items), "house": self.house_idx}


# ---------------------------------------------------------------------------
# 3 – Parameter bundle
# ---------------------------------------------------------------------------


@dataclass
class GenerationParams:
    houses_min: int
    houses_max: int
    categories_spec: Dict[str, List[str]]  # base pool for each category
    rng: random.Random

    k_between_range: Tuple[int, int] = (2, 2)  # default "exactly two houses between"
    max_constraints: int = 50  # safety valve to break dead loops

    allowed_rule_types: Sequence[type[Constraint]] = (
        #IsAt,
        NotAt,
        #SameHouse,
        DiffHouse,
        #Adjacent,
        NotAdjacent,
        #LeftOf,
        #RightOf,
        ExactlyKBetween,
        OneOfSetAt,
    )

    # ---- helpers --------------------------------------------------------
    def pick_house_count(self) -> int:
        return self.rng.randint(self.houses_min, self.houses_max)

    def build_categories(self, N: int) -> List[Category]:
        cats: List[Category] = []
        # Limit categories to at most N (number of houses)
        available_categories = list(self.categories_spec.keys())
        if len(available_categories) > N:
            # Randomly sample N categories
            chosen_categories = self.rng.sample(available_categories, N)
        else:
            # Use all available categories
            chosen_categories = available_categories
            
        for cname in chosen_categories:
            pool = self.categories_spec[cname]
            if len(pool) < N:
                raise ValueError(
                    f"Category '{cname}' only has {len(pool)} items but need ≥{N}."
                )
            chosen = self.rng.sample(pool, N) if len(pool) > N else list(pool)
            cats.append(Category(cname, [Item(name, cname) for name in chosen]))
        return cats


# ---------------------------------------------------------------------------
# 4 – Puzzle container + pretty helpers
# ---------------------------------------------------------------------------


@dataclass
class Puzzle:
    houses: List[House]
    categories: List[Category]
    constraints: List[Constraint]
    solution: Dict[int, Dict[str, Item]]  # answer key (always complete)
    target_category: str

    # ----- convenience renderers ---------------------------------------
    def english_clues(self) -> List[str]:
        return [c.to_text() for c in self.constraints]

    def solution_order(self) -> List[str]:
        """Return the list of *target category* item names in house order."""
        return [self.solution[h.index][self.target_category].name for h in self.houses]

    def complete_puzzle_description(self) -> str:
        """Generate the complete English description of the puzzle."""
        lines: List[str] = []
        
        # Add puzzle explanation
        lines.append("This is a logic puzzle where you need to figure out which items from")
        lines.append("different categories are in each house. Houses are arranged in a row")
        lines.append(f"and numbered from left to right from 1 to {len(self.houses)}.")
        lines.append("You are given a set of clues that describe relationships between")
        lines.append("the items and their locations. Your goal is to determine the")
        lines.append(f"correct order of {self.target_category.lower()} across all houses.")
        lines.append("")
        
        # List all categories and their items
        lines.append("Available Options:")
        for category in self.categories:
            items_list = ", ".join(item.name for item in category.items)
            lines.append(f"{category.name}: {items_list}")
        lines.append("")
        
        lines.append(f"Target category: {self.target_category}")
        lines.append("")

        lines.append("Clues:")
        for i, clue in enumerate(self.english_clues(), 1):
            lines.append(f"{i}. {clue}")
        lines.append("")
        lines.append("Goal:")
        lines.append(
            f"Work out which {self.target_category[:-1].lower() if self.target_category.endswith('s') else self.target_category.lower()} "
            f"is in each house (houses are numbered 1 left‑to‑right)."
            "You can think step-by-step as long as you want, but eventually you need to give your answer."
            "Give you answer inside a \\boxed{...} tag, and format your answer as a Python list."
            "So the answer should be like this: \\boxed{['item1', 'item2', 'item3', ...]}."
            "Remember, you only need to find the order of the items in the target category, not ALL items."
        )
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5 – Generator core
# ---------------------------------------------------------------------------


def generate_puzzle(params: GenerationParams) -> Puzzle:
    rng = params.rng

    # 1 – basic skeleton ---------------------------------------------------
    N = params.pick_house_count()
    houses = [House(i) for i in range(1, N + 1)]
    categories = params.build_categories(N)
    target_category = rng.choice(categories).name

    # 2 – random full solution -------------------------------------------
    solution: Dict[int, Dict[str, Item]] = {h.index: {} for h in houses}
    for cat in categories:
        items = cat.items[:]
        rng.shuffle(items)
        for h, itm in zip(houses, items):
            solution[h.index][cat.name] = itm

    # 3 – build minimal‑redundant clue list ------------------------------
    constraints: List[Constraint] = []
    attempts = 0
    while True:
        if _target_unique(constraints, categories, houses, target_category):
            # Already uniquely determined – try to drop any redundant rules.
            _minimize(constraints, categories, houses, target_category)
            break
        if attempts >= params.max_constraints:
            # Safety: restart generation from scratch.
            return generate_puzzle(params)
        new_rule = _fabricate_rule(rng, params, solution)
        if new_rule and _is_new(new_rule, constraints):
            constraints.append(new_rule)
            attempts = 0  # reset counter each time we actually add
        else:
            attempts += 1

    return Puzzle(houses, categories, constraints, solution, target_category)


# ---------------------------------------------------------------------------
# 5.1 – Rule fabrication helpers
# ---------------------------------------------------------------------------

def _fabricate_rule(rng: random.Random, params: GenerationParams, solution: Dict[int, Dict[str, Item]]) -> Optional[Constraint]:
    """Return a *true* rule that holds in *solution* (or None on rare failure)."""

    Rule = rng.choice(params.allowed_rule_types)
    houses = list(solution.keys())

    # helper to pick random items
    all_items = [itm for house in solution.values() for itm in house.values()]
    def pick_item() -> Item:
        return rng.choice(all_items)

    # map each Rule type to construction logic ---------------------------
    if Rule is IsAt:
        itm = pick_item()
        return IsAt(itm.name, _loc(itm.name, solution))

    if Rule is NotAt:
        itm = pick_item()
        loc = _loc(itm.name, solution)
        other = rng.choice([h for h in houses if h != loc])
        return NotAt(itm.name, other)

    if Rule in (SameHouse, DiffHouse, Adjacent, NotAdjacent, LeftOf, RightOf):
        itm_a, itm_b = rng.sample(all_items, 2)
        if Rule is SameHouse:
            if _loc(itm_a.name, solution) != _loc(itm_b.name, solution):
                # ensure they share house, pick another pair
                return None
            return SameHouse(itm_a.name, itm_b.name)
        if Rule is DiffHouse:
            if _loc(itm_a.name, solution) == _loc(itm_b.name, solution):
                return None
            return DiffHouse(itm_a.name, itm_b.name)
        if Rule is Adjacent:
            if abs(_loc(itm_a.name, solution) - _loc(itm_b.name, solution)) != 1:
                return None
            return Adjacent(itm_a.name, itm_b.name)
        if Rule is NotAdjacent:
            if abs(_loc(itm_a.name, solution) - _loc(itm_b.name, solution)) == 1:
                return None
            return NotAdjacent(itm_a.name, itm_b.name)
        if Rule is LeftOf:
            if _loc(itm_a.name, solution) >= _loc(itm_b.name, solution):
                return None
            return LeftOf(itm_a.name, itm_b.name)
        if Rule is RightOf:
            if _loc(itm_a.name, solution) <= _loc(itm_b.name, solution):
                return None
            return RightOf(itm_a.name, itm_b.name)

    if Rule is ExactlyKBetween:
        itm_a, itm_b = rng.sample(all_items, 2)
        diff = abs(_loc(itm_a.name, solution) - _loc(itm_b.name, solution)) - 1
        if diff < params.k_between_range[0] or diff > params.k_between_range[1]:
            return None
        return ExactlyKBetween(itm_a.name, itm_b.name, diff)

    if Rule is OneOfSetAt:
        itm_a, itm_b, itm_c = rng.sample(all_items, 3)
        # choose the house of exactly ONE of them
        chosen_itm = rng.choice([itm_a, itm_b, itm_c])
        h_idx = _loc(chosen_itm.name, solution)
        # ensure other two are elsewhere
        if any(_loc(itm.name, solution) == h_idx for itm in (itm_a, itm_b, itm_c) if itm != chosen_itm):
            return None
        return OneOfSetAt((itm_a.name, itm_b.name, itm_c.name), h_idx)

    return None  # fallback: give up this round


def _loc(item_name: str, sol: Dict[int, Dict[str, Item]]) -> int:
    for h, m in sol.items():
        for itm in m.values():
            if itm.name == item_name:
                return h
    raise ValueError("item not found in solution – should never happen")


# ---------------------------------------------------------------------------
# 5.2 – Uniqueness & search
# ---------------------------------------------------------------------------


def _target_unique(
    constraints: List[Constraint],
    categories: List[Category],
    houses: List[House],
    target_cat: str,
) -> bool:
    """Return **True** iff constraints force *one* ordering of *target_cat* items."""

    found_orders: List[Tuple[str, ...]] = []

    # backtracking -------------------------------------------------------
    def dfs(cat_idx: int, assignment: Dict[int, Dict[str, Item]]):
        nonlocal found_orders
        if len(found_orders) > 1:
            return  # already non‑unique – prune
        if cat_idx == len(categories):
            order = tuple(assignment[h.index][target_cat].name for h in houses)
            if order not in found_orders:
                found_orders.append(order)
            return
        cat = categories[cat_idx]
        for perm in permutations(cat.items):
            # assign this perm into houses
            for h, itm in zip(houses, perm):
                assignment.setdefault(h.index, {})[cat.name] = itm
            if all(c.is_satisfied(assignment) for c in constraints):
                dfs(cat_idx + 1, assignment)
            # un‑assign
            for h in houses:
                assignment[h.index].pop(cat.name)

    dfs(0, {h.index: {} for h in houses})
    return len(found_orders) == 1


# ---------------------------------------------------------------------------
# 5.3 – Constraint minimisation & utils
# ---------------------------------------------------------------------------

def _minimize(constraints: List[Constraint], categories: List[Category], houses: List[House], target_cat: str) -> None:
    """Greedy remove‑any‑that‑stay‑unique."""
    for c in constraints[:]:
        temp = [x for x in constraints if x is not c]
        if _target_unique(temp, categories, houses, target_cat):
            constraints.remove(c)


def _is_new(rule: Constraint, existing: List[Constraint]) -> bool:
    return all(rule.to_json() != ex.to_json() for ex in existing)


# ---------------------------------------------------------------------------
# 6 – Writers: JSON + text
# ---------------------------------------------------------------------------


def write_json(puzzle: Puzzle, path: Path) -> None:
    cats_dict = {
        cat.name: [itm.name for itm in cat.items] for cat in puzzle.categories
    }
    clues_json = [c.to_json() for c in puzzle.constraints]
    solution_json: Dict[str, List[str]] = {
        cat.name: [puzzle.solution[h.index][cat.name].name for h in puzzle.houses]
        for cat in puzzle.categories
    }
    data = {
        "houses": len(puzzle.houses),
        "categories": cats_dict,
        "target_category": puzzle.target_category,
        "clues": clues_json,
        "solution": solution_json,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf8")


def write_json_all(puzzles: List[Puzzle], path: Path) -> None:
    """Write all puzzles to a single JSON file."""
    puzzles_data = []
    for puzzle in puzzles:
        cats_dict = {
            cat.name: [itm.name for itm in cat.items] for cat in puzzle.categories
        }
        clues_json = [c.to_json() for c in puzzle.constraints]
        solution_json: Dict[str, List[str]] = {
            cat.name: [puzzle.solution[h.index][cat.name].name for h in puzzle.houses]
            for cat in puzzle.categories
        }
        data = {
            "metadata": {
                "houses": len(puzzle.houses),
                "categories": cats_dict,
                "target_category": puzzle.target_category,
                "clues": clues_json,
                "solution": solution_json,
            },
            "question": puzzle.complete_puzzle_description(),
            "canonical_answer": json.dumps(solution_json[puzzle.target_category])
        }
        puzzles_data.append(data)
    
    path.write_text(json.dumps(puzzles_data, indent=2), encoding="utf8")


STAR = "★"


def write_text(puzzle: Puzzle, path: Path, puzzle_no: int) -> None:
    lines: List[str] = []
    lines.append(f"ZebraLogic #{puzzle_no} — {len(puzzle.houses)} houses")
    lines.append("")
    lines.append(f"Target category: {puzzle.target_category}")
    lines.append("")

    lines.append("Clues")
    lines.append("-----")
    for i, clue in enumerate(puzzle.english_clues(), 1):
        lines.append(f"{i}. {clue}")
    lines.append("")
    lines.append("Goal")
    lines.append("----")
    lines.append(
        f"Work out which {puzzle.target_category[:-1].lower() if puzzle.target_category.endswith('s') else puzzle.target_category.lower()} "
        f"is in each house (houses are numbered 1 left‑to‑right)."
    )
    lines.append("")
    lines.append("Answer key (scroll past if you don't want spoilers)")
    lines.append("=" * 60)
    lines.append(" ")
    lines.append(", ".join(puzzle.solution_order()))

    path.write_text("\n".join(lines), encoding="utf8")


def write_text_all(puzzles: List[Puzzle], path: Path) -> None:
    """Write all puzzles to a single text file."""
    lines: List[str] = []
    
    for idx, puzzle in enumerate(puzzles, 1):
        if idx > 1:
            lines.append("")
            lines.append("=" * 80)
            lines.append("")
        
        # Use the complete puzzle description from the JSON
        lines.append(f"ZebraLogic #{idx} — {len(puzzle.houses)} houses")
        lines.append("")
        lines.append(puzzle.complete_puzzle_description())
        lines.append("")
        lines.append("Answer key (scroll past if you don't want spoilers)")
        lines.append("=" * 60)
        lines.append(" ")
        lines.append(", ".join(puzzle.solution_order()))

    path.write_text("\n".join(lines), encoding="utf8")


# ---------------------------------------------------------------------------
# 7 – CLI glue
# ---------------------------------------------------------------------------

def _default_categories() -> Dict[str, List[str]]:
    return {
        "Pets": [
            "dog",
            "cat",
            "parrot",
            "zebra",
            "goldfish",
            "hamster",
        ],
        "Drinks": [
            "water",
            "tea",
            "coffee",
            "milk",
            "juice",
            "cola",
        ],
        "Instruments": [
            "guitar",
            "piano",
            "violin",
            "flute",
            "trumpet",
            "drums",
        ],
        "Books": [
            "Bible",
            "Koran",
            "Iliad",
            "Odyssey",
            "Aeneid",
            "Inferno",
            "Beowulf",
        ],
        "Plants": [
            "rose",
            "tulip",
            "daisy",
            "sunflower",
            "lily",
            "orchid",
            "cactus",
            "fern",
            "ivy",
            "carnation",
            "daffodil",
            "hibiscus",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate Zebra/Einstein logic puzzles (single‑category goal).")
    ap.add_argument("--houses-min", type=int, default=3, help="Minimum house count (inclusive)")
    ap.add_argument("--houses-max", type=int, default=3, help="Maximum house count (inclusive)")
    ap.add_argument("--count", type=int, default=100, help="How many puzzles to generate")
    ap.add_argument("--seed", type=int, default=None, help="Random‑seed for reproducibility")
    ap.add_argument("--categories-file", type=Path, help="JSON file with category→items mapping")
    ap.add_argument("--output", type=Path, default=Path("."), help="Directory for output files")

    args = ap.parse_args(argv)

    # derive house range --------------------------------------------------
    min_h, max_h = args.houses_min, args.houses_max
    if min_h < 2 or max_h < min_h:
        ap.error("Invalid house range")

    # categories ----------------------------------------------------------
    if args.categories_file:
        try:
            categories_spec = json.loads(args.categories_file.read_text(encoding="utf8"))
        except Exception as e:
            sys.exit(f"Failed to read categories file: {e}")
    else:
        categories_spec = _default_categories()

    # RNG -----------------------------------------------------------------
    rng = random.Random(args.seed)

    # output dir ----------------------------------------------------------
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    params = GenerationParams(min_h, max_h, categories_spec, rng)

    # Generate all puzzles first
    puzzles = []
    for idx in range(1, args.count + 1):
        puzzle = generate_puzzle(params)
        puzzles.append(puzzle)
        print(f"Generated puzzle {idx}")

    # Write all puzzles to single files
    json_path = out_dir / "zebra_puzzles.json"
    txt_path = out_dir / "zebra_puzzles.txt"
    write_json_all(puzzles, json_path)
    write_text_all(puzzles, txt_path)
    print(f"Created {json_path} and {txt_path} with {len(puzzles)} puzzles")


# ---------------------------------------------------------------------------
# 8 – Entry point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
