"""Omvandlar en bordslasning till anvandbara rad.

En medveten designbeslut: pott och insatser lases *inte* av skarmen. Siffror ar
den delen OCR ar samst pa, och det var precis dar det gamla forsoket havererade
— potten lastes som $52142 i en hand och $7247 i nasta. Istallet visas vilken
equity du behover for att syna olika insatsstorlekar. Du ser sjalv vad
motstandaren satsat; den jamforelsen gor du pa en sekund, och den blir aldrig
fel pa grund av en feltolkad siffra.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from strategy.engine import OPEN_RAISE_RANGES
from trainer.cards import RANKS, evaluate, hand_class, hand_notation
from trainer.cards import equity as calc_equity

POSITION_ORDER = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]

# Vanliga insatsstorlekar och vilken equity som kravs for att synen ska ga plus
BET_SIZES = [
    ("1/4 pott", 0.25),
    ("1/2 pott", 0.50),
    ("3/4 pott", 0.75),
    ("pott", 1.00),
]


@dataclass
class Advice:
    headline: str = ""
    equity: float = 0.0
    hand_name: str = ""
    draws: List[str] = field(default_factory=list)
    outs: int = 0
    lines: List[str] = field(default_factory=list)
    playable_from: List[str] = field(default_factory=list)
    warning: str = ""


def _rank_values(cards: List[str]) -> List[int]:
    return sorted(RANKS.index(c[0].upper()) for c in cards)


def find_draws(hero: List[str], board: List[str]) -> Tuple[List[str], int]:
    """Namnge drag och uppskatta antalet outs. Tomt pa river — inget kommer mer."""
    if not board or len(board) >= 5:
        return [], 0

    all_cards = hero + board
    draws: List[str] = []
    outs = 0

    # Flushdrag: fyra i samma farg, och vi maste sjalva bidra
    suit_counts = Counter(c[1].lower() for c in all_cards)
    hero_suits = {c[1].lower() for c in hero}
    for suit, count in suit_counts.items():
        if count == 4 and suit in hero_suits:
            draws.append("Flushdrag (9 outs)")
            outs += 9
        elif count >= 5 and suit in hero_suits:
            pass  # redan flush, inget drag

    # Stegdrag: fonster om fem intilliggande rankar med fyra tratt
    values = set(_rank_values(all_cards))
    hero_values = set(_rank_values(hero))
    if 12 in values:            # ess kan spela lagt for A2345
        values.add(-1)
    best_straight_draw = None
    for low in range(-1, 9):
        window = set(range(low, low + 5))
        present = window & values
        if len(present) == 4 and (window & hero_values):
            missing = (window - present).pop()
            # Oppet i bada andar om de fyra vi har sitter i foljd
            consecutive = max(present) - min(present) == 3
            kind = "Oppet stegdrag (8 outs)" if consecutive else "Gutshot (4 outs)"
            if best_straight_draw is None or "Oppet" in kind:
                best_straight_draw = kind
    if best_straight_draw:
        draws.append(best_straight_draw)
        outs += 8 if "Oppet" in best_straight_draw else 4

    return draws, outs


def preflop_positions(notation: str) -> List[str]:
    """Fran vilka positioner ar handen vard att oppna?

    Battre an att forsoka lista ut din position fran skarmen: du vet redan var
    du sitter, och listan sager direkt om handen duger darifran.
    """
    return [pos for pos in POSITION_ORDER
            if notation in OPEN_RAISE_RANGES.get(pos, set())]


def describe_made_hand(hero: List[str], board: List[str]) -> str:
    if len(board) < 3:
        return ""
    return hand_class(evaluate(hero + board))


def build_advice(
    hero: List[str],
    board: List[str],
    opponents: int = 1,
    sims: int = 8000,
) -> Advice:
    """Rad utifran korten pa skarmen."""
    advice = Advice()

    if len(hero) != 2:
        advice.headline = "Vantar pa kort"
        return advice

    opponents = max(1, opponents)
    equity = calc_equity(hero, board, opponents, sims=sims)
    advice.equity = equity
    advice.hand_name = describe_made_hand(hero, board)
    advice.draws, advice.outs = find_draws(hero, board)

    notation = hand_notation(*hero)

    if not board:
        positions = preflop_positions(notation)
        advice.playable_from = positions
        if positions:
            advice.headline = f"{notation} — spelbar fran {', '.join(positions)}"
        else:
            advice.headline = f"{notation} — utanfor oppningsrange, folda"
        advice.lines.append(
            f"Equity mot {opponents} motstandare: {equity:.0%}"
        )
        if not positions:
            advice.lines.append(
                "Handen ar inte lonsam att oppna fran nagon position."
            )
        elif len(positions) == len(POSITION_ORDER):
            advice.lines.append("Premiumhand — hoj fran alla positioner.")
        else:
            earliest = POSITION_ORDER.index(positions[0])
            too_early = POSITION_ORDER[:earliest]
            advice.lines.append(
                f"Fran {', '.join(too_early)} — folda."
            )
        return advice

    # Postflop
    if equity >= 0.70:
        advice.headline = "Stark — satsa for varde"
    elif equity >= 0.55:
        advice.headline = "Bra — satsa eller syna"
    elif equity >= 0.40:
        advice.headline = "Medel — kontrollera potten"
    elif advice.outs:
        advice.headline = "Drag — spela pa odds"
    else:
        advice.headline = "Svag — checka eller folda"

    advice.lines.append(f"Equity mot {opponents} motstandare: {equity:.0%}")
    if advice.hand_name:
        advice.lines.append(f"Din hand: {advice.hand_name}")
    for draw in advice.draws:
        advice.lines.append(f"Drag: {draw}")

    advice.lines.append("Behover for att syna:")
    for label, fraction in BET_SIZES:
        needed = fraction / (1.0 + 2.0 * fraction)
        verdict = "syn" if equity > needed else "folda"
        advice.lines.append(
            f"   {label:<9} {needed:.0%}  ->  {verdict}"
        )

    return advice
