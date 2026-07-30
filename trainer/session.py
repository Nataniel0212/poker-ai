"""Sessionsstatistik, motstandarlasningar och lackrapport.

Direktfeedbacken larr dig ratt beslut i stunden. Den har modulen svarar pa den
andra fragan: *vilket* fel gor du om och om igen? Det ar dar den verkliga
forbattringen ligger.
"""

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from trainer.coach import Feedback


@dataclass
class VillainProfile:
    """Latt profil som matchar det strategy/engine.py:s exploit-kod laser."""
    name: str
    hands_played: int = 0
    vpip: float = 25.0
    pfr: float = 18.0
    aggression_factor: float = 2.0
    three_bet: float = 6.0
    fold_to_three_bet: float = 55.0
    fold_to_cbet: float = 50.0
    wtsd: float = 26.0
    player_type: str = "unknown"

    # ra raknare
    _hands: int = 0
    _vpip_hits: int = 0
    _pfr_hits: int = 0
    _bets: int = 0
    _calls: int = 0
    _folds_postflop: int = 0
    _showdowns: int = 0

    def note_hand(self, voluntarily_in: bool, raised_preflop: bool) -> None:
        self._hands += 1
        if voluntarily_in:
            self._vpip_hits += 1
        if raised_preflop:
            self._pfr_hits += 1
        self._recompute()

    def note_action(self, kind: str, street: str) -> None:
        if kind in ("bet", "raise", "all_in"):
            self._bets += 1
        elif kind == "call":
            self._calls += 1
        elif kind == "fold" and street != "preflop":
            self._folds_postflop += 1

    def note_showdown(self) -> None:
        self._showdowns += 1
        self._recompute()

    def _recompute(self) -> None:
        self.hands_played = self._hands
        if self._hands:
            self.vpip = 100.0 * self._vpip_hits / self._hands
            self.pfr = 100.0 * self._pfr_hits / self._hands
            self.wtsd = 100.0 * self._showdowns / self._hands
        if self._calls:
            self.aggression_factor = self._bets / self._calls
        elif self._bets:
            self.aggression_factor = 10.0

        total_post = self._folds_postflop + self._calls + self._bets
        if total_post:
            self.fold_to_cbet = 100.0 * self._folds_postflop / total_post

        self.player_type = self._classify()

    def _classify(self) -> str:
        if self._hands < 8:
            return "unknown"
        if self.vpip > 45 and self.aggression_factor < 1.0:
            return "calling_station"
        if self.vpip > 45 and self.aggression_factor > 3.0:
            return "maniac"
        if self.vpip < 18:
            return "nit"
        if self.vpip > 30:
            return "lag"
        return "tag"

    def read(self) -> str:
        """En mening om hur den har motstandaren spelar."""
        if self._hands < 8:
            return f"{self.name}: för få händer för en läsning ({self._hands})"
        labels = {
            "calling_station": "synar för mycket — bluffa aldrig, värdesatsa tunt",
            "maniac": "höjer med allt — låt hen bluffa in i dina starka händer",
            "nit": "spelar bara premium — stjäl blinds, respektera hens satsningar",
            "lag": "spelar brett och aggressivt — syna lättare",
            "tag": "solid — inga uppenbara hål att utnyttja",
            "unknown": "oklar stil än",
        }
        return (
            f"{self.name}: VPIP {self.vpip:.0f}% / AF {self.aggression_factor:.1f} — "
            f"{labels.get(self.player_type, '')}"
        )


@dataclass
class SessionStats:
    hands: int = 0
    decisions: int = 0
    correct: int = 0
    near: int = 0
    mistakes: int = 0
    ev_lost_bb: float = 0.0
    chips_start: float = 0.0
    chips_now: float = 0.0
    by_category: Counter = field(default_factory=Counter)
    ev_by_category: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_street: Counter = field(default_factory=Counter)
    correct_by_street: Counter = field(default_factory=Counter)

    def record(self, fb: Feedback) -> None:
        self.decisions += 1
        self.by_street[fb.street] += 1
        if fb.verdict == "ratt":
            self.correct += 1
            self.correct_by_street[fb.street] += 1
        elif fb.verdict == "narapa":
            self.near += 1
        else:
            self.mistakes += 1
        self.ev_lost_bb += fb.ev_loss_bb
        if fb.category:
            self.by_category[fb.category] += 1
            self.ev_by_category[fb.category] += fb.ev_loss_bb

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.decisions if self.decisions else 0.0

    def summary_lines(self, bb: float) -> List[str]:
        lines = [
            f"Händer spelade:      {self.hands}",
            f"Beslut:              {self.decisions}",
            f"Rätt:                {self.correct} ({self.accuracy:.0f}%)",
            f"Nästan:              {self.near}",
            f"Misstag:             {self.mistakes}",
        ]
        if self.decisions:
            lines.append(f"EV förlorat:         {self.ev_lost_bb:.1f}bb "
                         f"({self.ev_lost_bb / max(1, self.hands):.2f}bb/hand)")
        net = self.chips_now - self.chips_start
        if bb:
            lines.append(f"Resultat:            {net:+.0f} chips ({net / bb:+.1f}bb)")

        if self.by_street:
            lines.append("")
            lines.append("Träffsäkerhet per gata:")
            for street in ("preflop", "flop", "turn", "river"):
                total = self.by_street.get(street, 0)
                if total:
                    ok = self.correct_by_street.get(street, 0)
                    lines.append(f"   {street:<10} {ok}/{total} ({100.0 * ok / total:.0f}%)")

        if self.by_category:
            lines.append("")
            lines.append("Dina största läckor:")
            ranked = sorted(
                self.by_category.items(),
                key=lambda kv: -self.ev_by_category[kv[0]],
            )
            for cat, count in ranked[:5]:
                lines.append(
                    f"   {cat} — {count} ggr, {self.ev_by_category[cat]:.1f}bb"
                )
        return lines


class OpponentTracker:
    """Bygger profiler pa botarna medan du spelar."""

    def __init__(self):
        self.profiles: Dict[str, VillainProfile] = {}

    def get(self, name: str) -> VillainProfile:
        if name not in self.profiles:
            self.profiles[name] = VillainProfile(name=name)
        return self.profiles[name]

    def observe_hand(self, hand, hero_name: str) -> None:
        """Uppdatera profiler efter avslutad hand."""
        preflop = [a for a in hand.actions if a.street == "preflop"]
        for player in hand.players:
            if player.name == hero_name:
                continue
            prof = self.get(player.name)
            mine = [a for a in preflop if a.player == player.name]
            voluntarily = any(a.kind in ("call", "bet", "raise", "all_in") for a in mine)
            raised = any(a.kind in ("raise", "all_in") for a in mine)
            prof.note_hand(voluntarily, raised)
            for a in hand.actions:
                if a.player == player.name:
                    prof.note_action(a.kind, a.street)
            if player.active and len(hand.board) == 5 and \
                    sum(1 for p in hand.players if p.active) > 1:
                prof.note_showdown()

    def main_villain(self, hand, hero_name: str) -> Optional[VillainProfile]:
        """Den aktiva motstandare vi har mest data pa — anvands for exploits."""
        candidates = [
            self.get(p.name) for p in hand.players
            if p.name != hero_name and p.active
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda p: p.hands_played)
        return best if best.hands_played >= 8 else None

    def reads(self) -> List[str]:
        return [p.read() for p in sorted(
            self.profiles.values(), key=lambda p: -p.hands_played
        )]


def save_session(path: str, stats: SessionStats, tracker: OpponentTracker) -> None:
    """Spara sessionen sa framsteg kan foljas over tid."""
    payload = {
        "hands": stats.hands,
        "decisions": stats.decisions,
        "correct": stats.correct,
        "near": stats.near,
        "mistakes": stats.mistakes,
        "accuracy": round(stats.accuracy, 1),
        "ev_lost_bb": round(stats.ev_lost_bb, 2),
        "net_chips": round(stats.chips_now - stats.chips_start, 2),
        "categories": dict(stats.by_category),
        "ev_by_category": {k: round(v, 2) for k, v in stats.ev_by_category.items()},
    }
    history = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except (ValueError, OSError):
            history = []
    history.append(payload)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
