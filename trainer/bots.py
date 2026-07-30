"""Botmotstandare med olika spelstilar.

Poangen med skilda stilar ar att traningen ska innehalla de spelartyper du
faktiskt moter: nitar som bara spelar premium, stationer som synar allt, och
maniacs som hojer for mycket. Att lara sig kanna igen dem ar halva spelet.
"""

import random
from dataclasses import dataclass
from typing import Tuple

from trainer.cards import equity, rank_value, suit_of
from trainer.table import Hand, Options, Player


@dataclass
class Style:
    label: str
    description: str
    open_threshold: int       # lagsta Chen-poang for att oppna potten
    call_threshold: int       # lagsta Chen-poang for att syna en hojning
    aggression: float         # sannolikhet att hoja istallet for syna med stark hand
    bluff_freq: float         # sannolikhet att bluffa med svag hand
    station: float            # extra benagenhet att syna postflop (sanker foldgransen)


STYLES = {
    "nit": Style(
        "Nit", "Spelar bara premiumhander, foldar for mycket",
        open_threshold=10, call_threshold=11, aggression=0.35, bluff_freq=0.03, station=-0.05,
    ),
    "tag": Style(
        "TAG", "Tight-aggressiv, solid standardspelare",
        open_threshold=8, call_threshold=9, aggression=0.65, bluff_freq=0.18, station=0.0,
    ),
    "lag": Style(
        "LAG", "Loose-aggressiv, pressar mycket",
        open_threshold=6, call_threshold=7, aggression=0.80, bluff_freq=0.32, station=0.03,
    ),
    "station": Style(
        "Station", "Synar nastan allt, hojer nastan aldrig",
        open_threshold=7, call_threshold=4, aggression=0.15, bluff_freq=0.02, station=0.18,
    ),
    "maniac": Style(
        "Maniac", "Hojer med vad som helst, svar att lasa",
        open_threshold=4, call_threshold=5, aggression=0.90, bluff_freq=0.55, station=0.10,
    ),
}

BOT_NAMES = ["Erik", "Sofia", "Marcus", "Linnea", "Jonas", "Amanda", "Petter", "Hanna"]


def chen_score(card1: str, card2: str) -> float:
    """Chen-formeln — snabb och valetablerad styrkeuppskattning preflop."""
    values = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}  # A, K, Q, J

    def base(card: str) -> float:
        rv = rank_value(card)
        return values.get(rv, (rv + 2) / 2.0)

    hi, lo = (card1, card2) if rank_value(card1) >= rank_value(card2) else (card2, card1)
    score = base(hi)

    if rank_value(hi) == rank_value(lo):
        score = max(5.0, score * 2)
        return score

    if suit_of(hi) == suit_of(lo):
        score += 2

    gap = rank_value(hi) - rank_value(lo) - 1
    penalty = {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    score -= penalty

    if gap <= 1 and rank_value(hi) < 10:  # bada under Q
        score += 1

    return score


class Bot:
    """Beslutslogik for en botspelare."""

    def __init__(self, style_key: str, rng: random.Random = None):
        self.style_key = style_key
        self.style = STYLES[style_key]
        self.rng = rng or random.Random()

    def decide(self, hand: Hand, me: Player, opts: Options) -> Tuple[str, float]:
        if hand.street == "preflop":
            return self._preflop(hand, me, opts)
        return self._postflop(hand, me, opts)

    # ---------- preflop ----------

    def _preflop(self, hand: Hand, me: Player, opts: Options) -> Tuple[str, float]:
        score = chen_score(*me.hole)
        st = self.style
        facing_raise = opts.current_bet > hand.bb + 1e-9

        if facing_raise:
            n_raises = sum(
                1 for a in hand.actions_this_street() if a["action"] in ("raise", "all_in")
            )
            need = st.call_threshold + (2 if n_raises >= 2 else 0)
            if score < need:
                return ("fold", 0.0) if opts.call_amount > 0 else ("check", 0.0)
            # Stark nog — ibland 3-bet
            if score >= need + 4 and opts.can_raise and self.rng.random() < st.aggression:
                return ("raise", self._raise_to(hand, opts, 3.0))
            return ("call", 0.0)

        # Ingen hojning an — oppna eller limpa in
        if score < st.open_threshold:
            if opts.can_check:
                return ("check", 0.0)
            # BB-rabatt: station/maniac betalar garna sma belopp
            if opts.call_amount <= hand.bb * 0.5 and self.rng.random() < 0.5 + st.station:
                return ("call", 0.0)
            return ("fold", 0.0)

        if opts.can_raise and self.rng.random() < st.aggression + 0.15:
            return ("raise", self._raise_to(hand, opts, 2.5))
        return ("call", 0.0) if opts.call_amount > 0 else ("check", 0.0)

    def _raise_to(self, hand: Hand, opts: Options, multiplier: float) -> float:
        target = max(hand.bb * multiplier, opts.current_bet * multiplier)
        target = max(target, opts.min_raise_to)
        return min(target, opts.max_raise_to)

    # ---------- postflop ----------

    def _postflop(self, hand: Hand, me: Player, opts: Options) -> Tuple[str, float]:
        st = self.style
        n_opp = max(1, sum(1 for p in hand.players if p.active) - 1)
        eq = equity(me.hole, hand.board, n_opp, sims=500, rng=self.rng)

        pot = opts.pot
        to_call = opts.call_amount

        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
            threshold = pot_odds - st.station

            if eq > 0.70 and opts.can_raise and self.rng.random() < st.aggression:
                return ("raise", self._bet_to(hand, opts, 0.75))
            if eq >= threshold:
                return ("call", 0.0)
            # Bluffhojning ibland
            if opts.can_raise and self.rng.random() < st.bluff_freq * 0.4:
                return ("raise", self._bet_to(hand, opts, 0.8))
            return ("fold", 0.0)

        # Ingen bet mot oss — beta eller checka
        if eq > 0.62 and opts.can_raise and self.rng.random() < st.aggression + 0.2:
            return ("raise", self._bet_to(hand, opts, 0.66))
        if eq < 0.35 and opts.can_raise and self.rng.random() < st.bluff_freq:
            return ("raise", self._bet_to(hand, opts, 0.5))
        if opts.can_check:
            return ("check", 0.0)
        return ("fold", 0.0)

    def _bet_to(self, hand: Hand, opts: Options, pot_fraction: float) -> float:
        target = opts.current_bet + max(hand.bb, opts.pot * pot_fraction)
        target = max(target, opts.min_raise_to)
        return min(target, opts.max_raise_to)


def make_table(
    hero_name: str = "Du",
    starting_stack: float = 10000.0,
    num_players: int = 6,
    rng: random.Random = None,
    styles: list = None,
) -> Tuple[list, dict]:
    """Skapa spelare + botar. Returnerar (players, bots-by-name)."""
    rng = rng or random.Random()
    style_keys = styles or ["tag", "station", "nit", "lag", "maniac", "tag"]

    players = [Player(hero_name, starting_stack, is_hero=True, style="hero")]
    names = BOT_NAMES[:]
    rng.shuffle(names)

    bots = {}
    for i in range(num_players - 1):
        key = style_keys[i % len(style_keys)]
        name = names[i]
        players.append(Player(name, starting_stack, style=key))
        bots[name] = Bot(key, rng)

    return players, bots
