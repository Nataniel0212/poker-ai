"""No-Limit Texas Hold'em-motor for traningsspelet.

Motorn ager hela speltillstandet: varje kort, varje stack och varje insats ar
exakt kant. Det ar hela poangen — ingen OCR, inga gissningar, ingen fordrojning.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from trainer.cards import Deck, evaluate, hand_class

STREETS = ("preflop", "flop", "turn", "river", "showdown")

# Positionsnamn raknat fran small blind och runt bordet.
_POSITIONS_FROM_SB = {
    2: ["SB", "BB"],
    3: ["SB", "BB", "BTN"],
    4: ["SB", "BB", "CO", "BTN"],
    5: ["SB", "BB", "HJ", "CO", "BTN"],
    6: ["SB", "BB", "UTG", "HJ", "CO", "BTN"],
}


@dataclass
class Player:
    name: str
    stack: float
    is_hero: bool = False
    style: str = "tag"
    hole: List[str] = field(default_factory=list)
    position: str = ""
    folded: bool = False
    all_in: bool = False
    street_bet: float = 0.0   # insatt denna street
    total_bet: float = 0.0    # insatt hela handen
    won: float = 0.0

    @property
    def active(self) -> bool:
        """Kvar i handen (kan fortfarande vinna potten)."""
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in and self.stack > 0


@dataclass
class Action:
    player: str
    kind: str          # fold, check, call, bet, raise, all_in
    amount: float = 0.0     # chips som lades till potten
    to_amount: float = 0.0  # total street_bet efter handlingen
    street: str = "preflop"


@dataclass
class Options:
    """Vad spelaren i tur far gora just nu."""
    can_fold: bool
    can_check: bool
    call_amount: float          # chips att lagga till for att syna (0 = check)
    can_raise: bool
    min_raise_to: float         # lagsta tillatna nya street_bet
    max_raise_to: float         # hogsta (all-in)
    pot: float
    current_bet: float


class PotResult:
    def __init__(self, amount: float, winners: List[str], description: str):
        self.amount = amount
        self.winners = winners
        self.description = description


class Hand:
    """En enskild hand. Drivs framat genom att svara pa `next_to_act`."""

    def __init__(
        self,
        players: List[Player],
        button: int,
        small_blind: float,
        big_blind: float,
        rng: random.Random = None,
    ):
        if not 2 <= len(players) <= 6:
            raise ValueError("Stoder 2-6 spelare")

        self.players = players
        self.button = button % len(players)
        self.sb = small_blind
        self.bb = big_blind
        self.rng = rng or random.Random()
        self.deck = Deck(self.rng)

        self.board: List[str] = []
        self.street: str = "preflop"
        self.actions: List[Action] = []
        self.current_bet: float = 0.0
        self.last_raise_size: float = big_blind
        self.finished: bool = False
        self.results: List[PotResult] = []
        self.aggressor: Optional[int] = None  # senaste som betade/hojde

        self._acted: set = set()
        self._turn: Optional[int] = None

        self._assign_positions()
        self._deal_holes()
        self._post_blinds()
        self._turn = self._first_to_act_preflop()
        self._advance_if_no_action_needed()

    # ---------- uppsattning ----------

    def _assign_positions(self) -> None:
        n = len(self.players)
        names = _POSITIONS_FROM_SB[n]
        sb_index = self.button if n == 2 else (self.button + 1) % n
        for offset, pos in enumerate(names):
            self.players[(sb_index + offset) % n].position = pos

    def _deal_holes(self) -> None:
        for p in self.players:
            p.hole = self.deck.deal(2)
            p.folded = False
            p.all_in = False
            p.street_bet = 0.0
            p.total_bet = 0.0
            p.won = 0.0

    def _post_blinds(self) -> None:
        sb_player = self._by_position("SB")
        bb_player = self._by_position("BB")
        self._commit(sb_player, min(self.sb, sb_player.stack))
        self._commit(bb_player, min(self.bb, bb_player.stack))
        self.current_bet = max(p.street_bet for p in self.players)
        self.last_raise_size = self.bb

    def _by_position(self, pos: str) -> Player:
        for p in self.players:
            if p.position == pos:
                return p
        raise KeyError(pos)

    def _index_of(self, player: Player) -> int:
        return self.players.index(player)

    def _first_to_act_preflop(self) -> Optional[int]:
        n = len(self.players)
        bb_index = self._index_of(self._by_position("BB"))
        return self._next_actor((bb_index + 1) % n, inclusive=True)

    def _first_to_act_postflop(self) -> Optional[int]:
        n = len(self.players)
        sb_index = self._index_of(self._by_position("SB"))
        return self._next_actor(sb_index, inclusive=True)

    def _next_actor(self, start: int, inclusive: bool = False) -> Optional[int]:
        n = len(self.players)
        i = start % n
        for step in range(n):
            idx = (i + step) % n
            if step == 0 and not inclusive:
                continue
            if self.players[idx].can_act:
                return idx
        return None

    # ---------- pengar ----------

    def _commit(self, player: Player, amount: float) -> float:
        """Flytta chips fran stack till potten. Returnerar faktiskt belopp."""
        amount = max(0.0, min(amount, player.stack))
        player.stack -= amount
        player.street_bet += amount
        player.total_bet += amount
        if player.stack <= 1e-9:
            player.stack = 0.0
            player.all_in = True
        return amount

    @property
    def pot(self) -> float:
        """Hela potten inklusive insatser pa nuvarande street."""
        return sum(p.total_bet for p in self.players)

    @property
    def pot_before_street(self) -> float:
        return sum(p.total_bet - p.street_bet for p in self.players)

    # ---------- flode ----------

    @property
    def next_to_act(self) -> Optional[Player]:
        if self.finished or self._turn is None:
            return None
        return self.players[self._turn]

    def options(self) -> Optional[Options]:
        player = self.next_to_act
        if player is None:
            return None

        to_call = max(0.0, self.current_bet - player.street_bet)
        to_call = min(to_call, player.stack)
        can_check = self.current_bet - player.street_bet <= 1e-9

        # En hojning kraver att nagon annan fortfarande kan agera.
        others_live = [
            p for p in self.players
            if p is not player and not p.folded and not p.all_in
        ]
        max_raise_to = player.street_bet + player.stack
        min_raise_to = self.current_bet + self.last_raise_size
        can_raise = bool(others_live) and max_raise_to > self.current_bet + 1e-9
        if can_raise:
            min_raise_to = min(min_raise_to, max_raise_to)

        return Options(
            can_fold=not can_check or self.current_bet > 0,
            can_check=can_check,
            call_amount=to_call,
            can_raise=can_raise,
            min_raise_to=min_raise_to,
            max_raise_to=max_raise_to,
            pot=self.pot,
            current_bet=self.current_bet,
        )

    def act(self, kind: str, to_amount: float = 0.0) -> Action:
        """Utfor handlingen for spelaren i tur.

        kind: 'fold' | 'check' | 'call' | 'raise' (aven forsta bet) | 'all_in'
        to_amount: for 'raise' — total street_bet att ga till.
        """
        player = self.next_to_act
        if player is None:
            raise RuntimeError("Ingen spelare i tur")
        opts = self.options()
        idx = self._turn

        if kind == "fold":
            player.folded = True
            action = Action(player.name, "fold", 0.0, player.street_bet, self.street)

        elif kind == "check":
            if not opts.can_check:
                raise ValueError("Kan inte checka nar det finns en bet")
            action = Action(player.name, "check", 0.0, player.street_bet, self.street)

        elif kind == "call":
            paid = self._commit(player, opts.call_amount)
            action = Action(player.name, "call", paid, player.street_bet, self.street)

        elif kind in ("raise", "bet", "all_in"):
            if kind == "all_in":
                to_amount = opts.max_raise_to
            if to_amount >= opts.max_raise_to - 1e-9:
                to_amount = opts.max_raise_to          # all-in far understiga min-raise
            elif to_amount < opts.min_raise_to - 1e-9:
                raise ValueError(
                    f"Hojning maste vara minst {opts.min_raise_to:.0f}"
                )
            if to_amount <= self.current_bet + 1e-9:
                raise ValueError("Hojning maste overstiga nuvarande bet")

            raise_size = to_amount - self.current_bet
            paid = self._commit(player, to_amount - player.street_bet)
            # Ett all-in som ar mindre an en full hojning oppnar inte budgivningen igen,
            # men vi haller det enkelt: alla far agera igen anda.
            if raise_size >= self.last_raise_size - 1e-9:
                self.last_raise_size = raise_size
            self.current_bet = max(self.current_bet, player.street_bet)
            self.aggressor = idx
            label = "all_in" if player.all_in else ("bet" if opts.current_bet <= 1e-9 else "raise")
            action = Action(player.name, label, paid, player.street_bet, self.street)
            self._acted = set()  # alla andra maste svara pa hojningen

        else:
            raise ValueError(f"Okand handling: {kind}")

        self.actions.append(action)
        self._acted.add(idx)
        self._advance()
        return action

    def _advance(self) -> None:
        """Ga vidare till nasta spelare, nasta street, eller avsluta handen."""
        if self._only_one_left():
            self._finish_uncontested()
            return

        nxt = self._next_unsettled()
        if nxt is not None:
            self._turn = nxt
            return

        self._next_street()

    def _advance_if_no_action_needed(self) -> None:
        """Om ingen kan agera (t.ex. alla all-in via blinds) — spola fram."""
        if self._only_one_left():
            self._finish_uncontested()
            return
        if self._next_unsettled() is None:
            self._next_street()

    def _only_one_left(self) -> bool:
        return sum(1 for p in self.players if p.active) <= 1

    def _next_unsettled(self) -> Optional[int]:
        """Nasta spelare som fortfarande maste agera pa denna street."""
        live = [i for i, p in enumerate(self.players) if p.can_act]
        if not live:
            return None
        # Om bara en spelare kan agera och alla andra ar all-in/foldade
        # behover hen bara agera om hen inte matchat insatsen.
        start = self._turn if self._turn is not None else -1
        n = len(self.players)
        for step in range(1, n + 1):
            idx = (start + step) % n
            p = self.players[idx]
            if not p.can_act:
                continue
            unmatched = p.street_bet < self.current_bet - 1e-9
            if unmatched or idx not in self._acted:
                return idx
        return None

    def _next_street(self) -> None:
        for p in self.players:
            p.street_bet = 0.0
        self.current_bet = 0.0
        self.last_raise_size = self.bb
        self._acted = set()
        self.aggressor = None

        order = list(STREETS)
        nxt = order[order.index(self.street) + 1]

        if nxt == "flop":
            self.deck.burn()
            self.board += self.deck.deal(3)
        elif nxt in ("turn", "river"):
            self.deck.burn()
            self.board += self.deck.deal(1)

        self.street = nxt

        if nxt == "showdown":
            self._showdown()
            return

        # Om hogst en spelare kan agera (resten all-in) — spola vidare direkt.
        if sum(1 for p in self.players if p.can_act) <= 1:
            self._turn = None
            self._next_street()
            return

        self._turn = self._first_to_act_postflop()
        if self._turn is None:
            self._next_street()

    # ---------- avslut ----------

    def _finish_uncontested(self) -> None:
        winner = next(p for p in self.players if p.active)
        amount = self.pot
        winner.stack += amount
        winner.won = amount
        self.results = [PotResult(amount, [winner.name], "alla andra foldade")]
        self.street = "showdown"
        self.finished = True
        self._turn = None

    def _showdown(self) -> None:
        contenders = [p for p in self.players if p.active]
        if len(contenders) == 1:
            self._finish_uncontested()
            return

        while len(self.board) < 5:
            self.board += self.deck.deal(1)

        scores = {p.name: evaluate(p.hole + self.board) for p in contenders}
        self.results = []

        # Bygg huvud- och sidopotter utifran hur mycket var och en betalade in.
        levels = sorted({p.total_bet for p in self.players if p.total_bet > 0})
        prev = 0.0
        for level in levels:
            amount = sum(
                min(p.total_bet, level) - min(p.total_bet, prev)
                for p in self.players
            )
            eligible = [p for p in contenders if p.total_bet >= level - 1e-9]
            if amount <= 1e-9:
                prev = level
                continue
            if not eligible:
                eligible = contenders  # foldad spelare la in mest — gar till kvarvarande

            best = min(scores[p.name] for p in eligible)
            winners = [p for p in eligible if scores[p.name] == best]
            share = amount / len(winners)
            for w in winners:
                w.stack += share
                w.won += share
            self.results.append(
                PotResult(
                    amount,
                    [w.name for w in winners],
                    hand_class(best),
                )
            )
            prev = level

        self.finished = True
        self._turn = None

    # ---------- vy for strategimotorn ----------

    def actions_this_street(self) -> List[Dict]:
        return [
            {"player": a.player, "action": a.kind, "amount": a.amount}
            for a in self.actions
            if a.street == self.street
        ]

    def context_for(self, player: Player) -> Dict:
        """Bygg samma dict-format som strategy/engine.py forvantar sig."""
        villains = [
            {
                "name": p.name,
                "position": p.position,
                "stack": p.stack,
                "current_bet": p.street_bet,
            }
            for p in self.players
            if p is not player and p.active
        ]
        return {
            "hero_cards": player.hole,
            "hero_position": player.position,
            "hero_stack": player.stack,
            "community_cards": self.board,
            "pot": self.pot,
            "street": self.street,
            "big_blind": self.bb,
            "num_active_players": sum(1 for p in self.players if p.active),
            "villains": villains,
            "actions_this_street": self.actions_this_street(),
            "all_actions": [
                {
                    "player": a.player,
                    "action": a.kind,
                    "amount": a.amount,
                    "street": a.street,
                }
                for a in self.actions
            ],
            "is_tournament": False,
        }

    def showdown_summary(self) -> List[str]:
        """Radvis sammanfattning av vem som visade vad."""
        lines = []
        contenders = [p for p in self.players if p.active]
        if len(contenders) > 1 and len(self.board) == 5:
            for p in contenders:
                score = evaluate(p.hole + self.board)
                lines.append(f"{p.name}: {' '.join(p.hole)} — {hand_class(score)}")
        return lines


def play_blind_hand(
    players: List[Player],
    button: int,
    sb: float,
    bb: float,
    policy: Callable[[Hand, Player, Options], Sequence],
    rng: random.Random = None,
) -> Hand:
    """Kor en hel hand dar *alla* spelare styrs av `policy`. Anvands i tester."""
    hand = Hand(players, button, sb, bb, rng)
    guard = 0
    while not hand.finished:
        guard += 1
        if guard > 500:
            raise RuntimeError("Handen tog aldrig slut — bugg i budgivningen")
        player = hand.next_to_act
        opts = hand.options()
        kind, amount = policy(hand, player, opts)
        hand.act(kind, amount)
    return hand
