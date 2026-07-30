"""Kort, kortlek och equity-berakning.

Kort representeras som strangar i formatet 'As', 'Th', '7d' — samma format som
strategy/engine.py redan anvander, sa modulerna kan pratas vid utan konvertering.
"""

import random
from typing import Iterable, List, Sequence

from phevaluator import evaluate_cards

RANKS = "23456789TJQKA"
SUITS = "shdc"

FULL_DECK: List[str] = [r + s for r in RANKS for s in SUITS]

SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RED_SUITS = ("h", "d")

# Rank-namn for laslig utskrift av handstyrka
_HAND_CLASS_LIMITS = [
    (10, "Royal flush"),
    (166, "Straight flush"),
    (322, "Fyrtal"),
    (1599, "Kak"),
    (1609, "Flush"),
    (2467, "Stege"),
    (3325, "Triss"),
    (6185, "Tva par"),
    (6678, "Ett par"),
    (7462, "Hogt kort"),
]


def rank_of(card: str) -> str:
    return card[0].upper()


def suit_of(card: str) -> str:
    return card[1].lower()


def rank_value(card: str) -> int:
    """2 -> 0, A -> 12."""
    return RANKS.index(rank_of(card))


def pretty(card: str) -> str:
    """'As' -> 'A♠'."""
    return f"{rank_of(card)}{SUIT_SYMBOL[suit_of(card)]}"


def pretty_all(cards: Iterable[str]) -> str:
    return " ".join(pretty(c) for c in cards)


def hand_class(score: int) -> str:
    """Oversatt phevaluator-score till svenskt namn pa handkategorin."""
    for limit, name in _HAND_CLASS_LIMITS:
        if score <= limit:
            return name
    return "Okand"


def evaluate(cards: Sequence[str]) -> int:
    """Rankar 5-7 kort. Lagre varde = starkare hand (1 = royal flush)."""
    return evaluate_cards(*cards)


class Deck:
    """Blandad kortlek."""

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()
        self.cards: List[str] = FULL_DECK[:]
        self.rng.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[str]:
        if n > len(self.cards):
            raise ValueError("Kortleken ar slut")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def burn(self, n: int = 1) -> None:
        self.cards = self.cards[n:]

    def remaining(self) -> int:
        return len(self.cards)


def equity(
    hero: Sequence[str],
    board: Sequence[str],
    num_opponents: int = 1,
    sims: int = 4000,
    rng: random.Random = None,
) -> float:
    """Monte Carlo-equity mot slumpmassiga motstandarhander.

    Returnerar andelen av potten vi vinner i snitt (delade potter raknas som halva).
    Mot okanda rangen ar slumpmassiga hander ratt modell; nar vi vet mer om
    motstandaren snavas rangen till i coach.py.
    """
    rng = rng or random
    hero = list(hero)
    board = list(board)

    if len(hero) != 2:
        return 0.5
    num_opponents = max(1, num_opponents)

    known = set(hero) | set(board)
    if len(known) != len(hero) + len(board):
        return 0.5  # duplicerade kort — ska inte handa, men krascha inte

    deck = [c for c in FULL_DECK if c not in known]
    need_board = 5 - len(board)
    need = need_board + 2 * num_opponents
    if need > len(deck):
        return 0.5

    wins = 0.0
    for _ in range(sims):
        sample = rng.sample(deck, need)
        full_board = board + sample[:need_board]
        hero_score = evaluate(hero + full_board)

        idx = need_board
        best_opp = 9999
        for _ in range(num_opponents):
            opp = sample[idx:idx + 2]
            idx += 2
            score = evaluate(opp + full_board)
            if score < best_opp:
                best_opp = score

        if hero_score < best_opp:
            wins += 1.0
        elif hero_score == best_opp:
            wins += 0.5

    return wins / sims if sims else 0.5


def hand_notation(card1: str, card2: str) -> str:
    """('As', 'Kh') -> 'AKo'. ('7h', '7d') -> '77'."""
    r1, r2 = rank_of(card1), rank_of(card2)
    s1, s2 = suit_of(card1), suit_of(card2)

    if rank_value(card1) < rank_value(card2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    if r1 == r2:
        return r1 + r2
    return f"{r1}{r2}{'s' if s1 == s2 else 'o'}"
