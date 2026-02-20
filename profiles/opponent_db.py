"""
Opponent profiling system — tracks stats for every player encountered,
classifies their playing style, and provides exploit recommendations.
"""

import sqlite3
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlayerProfile:
    name: str
    hands_played: int = 0

    # Basic stats
    vpip: float = 0.0           # Voluntarily Put $ In Pot %
    pfr: float = 0.0            # Pre-Flop Raise %
    af: float = 0.0             # Aggression Factor (bet+raise / call)
    three_bet: float = 0.0      # 3-Bet %
    fold_to_three_bet: float = 0.0
    cbet: float = 0.0           # Continuation Bet %
    fold_to_cbet: float = 0.0
    wtsd: float = 0.0           # Went to Showdown %
    wsd: float = 0.0            # Won $ at Showdown %

    # Advanced stats
    check_raise_pct: float = 0.0
    donk_bet_pct: float = 0.0
    squeeze_pct: float = 0.0
    overbet_pct: float = 0.0

    # Bluff frequency per street
    bluff_flop: float = 0.0
    bluff_turn: float = 0.0
    bluff_river: float = 0.0

    # Raw counters for calculating percentages
    _times_could_vpip: int = 0
    _times_did_vpip: int = 0
    _times_could_pfr: int = 0
    _times_did_pfr: int = 0
    _times_could_3bet: int = 0
    _times_did_3bet: int = 0
    _times_faced_3bet: int = 0
    _times_folded_to_3bet: int = 0
    _times_could_cbet: int = 0
    _times_did_cbet: int = 0
    _times_faced_cbet: int = 0
    _times_folded_to_cbet: int = 0
    _total_bets_raises: int = 0
    _total_calls: int = 0
    _times_reached_showdown: int = 0
    _times_won_showdown: int = 0
    _times_could_check_raise: int = 0
    _times_did_check_raise: int = 0

    # Classification
    player_type: str = "unknown"  # fish, shark, maniac, nit, calling_station

    def recalculate(self):
        """Recalculate all percentage stats from raw counters."""
        if self._times_could_vpip > 0:
            self.vpip = (self._times_did_vpip / self._times_could_vpip) * 100
        if self._times_could_pfr > 0:
            self.pfr = (self._times_did_pfr / self._times_could_pfr) * 100
        if self._total_calls > 0:
            self.af = self._total_bets_raises / self._total_calls
        else:
            self.af = min(self._total_bets_raises, 10.0)  # Cap at 10 when no calls
        if self._times_could_3bet > 0:
            self.three_bet = (self._times_did_3bet / self._times_could_3bet) * 100
        if self._times_faced_3bet > 0:
            self.fold_to_three_bet = (self._times_folded_to_3bet / self._times_faced_3bet) * 100
        if self._times_could_cbet > 0:
            self.cbet = (self._times_did_cbet / self._times_could_cbet) * 100
        if self._times_faced_cbet > 0:
            self.fold_to_cbet = (self._times_folded_to_cbet / self._times_faced_cbet) * 100
        if self.hands_played > 0:
            self.wtsd = (self._times_reached_showdown / self.hands_played) * 100
        if self._times_reached_showdown > 0:
            self.wsd = (self._times_won_showdown / self._times_reached_showdown) * 100
        if self._times_could_check_raise > 0:
            self.check_raise_pct = (self._times_did_check_raise / self._times_could_check_raise) * 100

        self._classify()

    def _classify(self):
        """Classify player type based on stats. Requires 20+ hands for accuracy."""
        if self.hands_played < 20:
            self.player_type = "unknown"
            return

        # Fish: high VPIP (>40%), low PFR (<15%)
        if self.vpip > 40 and self.pfr < 15:
            self.player_type = "fish"
        # Calling station: high VPIP (>35%), low AF (<1.5)
        elif self.vpip > 35 and self.af < 1.5:
            self.player_type = "calling_station"
        # Maniac: high VPIP (>35%), high AF (>3)
        elif self.vpip > 35 and self.af > 3:
            self.player_type = "maniac"
        # Nit: very low VPIP (<18%), low PFR
        elif self.vpip < 18:
            self.player_type = "nit"
        # TAG (Tight Aggressive): VPIP 20-28%, PFR close to VPIP, AF > 2
        elif 20 <= self.vpip <= 28 and self.af > 2:
            self.player_type = "tag"
        # LAG (Loose Aggressive): VPIP 28-38%, AF > 2.5
        elif 28 < self.vpip <= 38 and self.af > 2.5:
            self.player_type = "lag"
        # Shark: balanced stats (hard to classify)
        elif 22 <= self.vpip <= 30 and 18 <= self.pfr <= 25 and 2 <= self.af <= 3.5:
            self.player_type = "shark"
        else:
            self.player_type = "regular"

    def get_exploit_tips(self) -> list:
        """Get specific exploit recommendations based on player type and stats."""
        tips = []

        if self.hands_played < 20:
            tips.append("Otillräcklig data (<20 händer) — spela GTO tills vi har mer info")
            return tips

        # Specific stat-based exploits
        if self.fold_to_cbet > 65:
            tips.append(f"C-bet brett — foldar till c-bet {self.fold_to_cbet:.0f}% av tiden")
        elif self.fold_to_cbet < 35:
            tips.append(f"Undvik att bluffa c-bet — callar c-bet {100-self.fold_to_cbet:.0f}%")

        if self.fold_to_three_bet > 70:
            tips.append(f"3-bet light — foldar till 3-bet {self.fold_to_three_bet:.0f}%")
        elif self.fold_to_three_bet < 40:
            tips.append(f"Bara value 3-bet — callar/4-bet {100-self.fold_to_three_bet:.0f}%")

        if self.vpip > 40:
            tips.append("Value bet tunt — spelar för många händer, callar med svaga holdings")
        if self.vpip < 18:
            tips.append("Stjäl blinds — spelar extremt tight, foldar det mesta")
            tips.append("Respektera deras raises — de har nästan alltid en stark hand")

        if self.af < 1.0 and self.hands_played > 30:
            tips.append("Bluffa ALDRIG — callar allt. Value bet stort med starka händer")
        elif self.af > 4.0:
            tips.append("Trappa — överaggressiv, callar ner med medium-starka händer")

        if self.wtsd > 35:
            tips.append("Bluffa mindre — går till showdown ofta, vill se kort")
        elif self.wtsd < 20:
            tips.append("Bluffa mer rivers — ger upp händer innan showdown")

        # Type-based general tips
        type_tips = {
            "fish": "Generellt: value bet brett, undvik fancy plays, håll det enkelt",
            "calling_station": "Generellt: value bet max, bluffa aldrig, bigger is better",
            "maniac": "Generellt: tighta upp, trap med monsters, låt dem hänga sig själva",
            "nit": "Generellt: stjäl, fold mot aggression, de har alltid det",
            "tag": "Generellt: svåraste motståndartypen, spela solid GTO",
            "lag": "Generellt: 3-bet mer, don't float utan equity, 4-bet light ibland",
            "shark": "Generellt: fullständig GTO — ingen tydlig svaghet att exploatera",
        }
        if self.player_type in type_tips:
            tips.append(type_tips[self.player_type])

        return tips

    def summary_string(self) -> str:
        """Human-readable summary of player stats."""
        type_icons = {
            "fish": "🐟", "shark": "🦈", "maniac": "🐒", "nit": "🪨",
            "calling_station": "📞", "tag": "🎯", "lag": "⚡",
            "regular": "👤", "unknown": "❓",
        }
        icon = type_icons.get(self.player_type, "❓")
        return (
            f"{icon} {self.player_type.upper()} ({self.hands_played} hands)\n"
            f"VPIP: {self.vpip:.1f}% | PFR: {self.pfr:.1f}% | AF: {self.af:.1f}\n"
            f"3-Bet: {self.three_bet:.1f}% | Fold to 3-Bet: {self.fold_to_three_bet:.1f}%\n"
            f"C-Bet: {self.cbet:.1f}% | Fold to C-Bet: {self.fold_to_cbet:.1f}%\n"
            f"WTSD: {self.wtsd:.1f}% | W$SD: {self.wsd:.1f}%"
        )


class OpponentDatabase:
    """SQLite-backed opponent profiling database."""

    def __init__(self, db_path: str = "opponents.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._cache: dict[str, PlayerProfile] = {}

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS players (
                name TEXT PRIMARY KEY,
                hands_played INTEGER DEFAULT 0,
                times_could_vpip INTEGER DEFAULT 0,
                times_did_vpip INTEGER DEFAULT 0,
                times_could_pfr INTEGER DEFAULT 0,
                times_did_pfr INTEGER DEFAULT 0,
                times_could_3bet INTEGER DEFAULT 0,
                times_did_3bet INTEGER DEFAULT 0,
                times_faced_3bet INTEGER DEFAULT 0,
                times_folded_to_3bet INTEGER DEFAULT 0,
                times_could_cbet INTEGER DEFAULT 0,
                times_did_cbet INTEGER DEFAULT 0,
                times_faced_cbet INTEGER DEFAULT 0,
                times_folded_to_cbet INTEGER DEFAULT 0,
                total_bets_raises INTEGER DEFAULT 0,
                total_calls INTEGER DEFAULT 0,
                times_reached_showdown INTEGER DEFAULT 0,
                times_won_showdown INTEGER DEFAULT 0,
                times_could_check_raise INTEGER DEFAULT 0,
                times_did_check_raise INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def get_profile(self, name: str) -> PlayerProfile:
        """Get or create a player profile."""
        if name in self._cache:
            return self._cache[name]

        row = self.conn.execute(
            "SELECT * FROM players WHERE name = ?", (name,)
        ).fetchone()

        profile = PlayerProfile(name=name)

        if row:
            profile.hands_played = row[1]
            profile._times_could_vpip = row[2]
            profile._times_did_vpip = row[3]
            profile._times_could_pfr = row[4]
            profile._times_did_pfr = row[5]
            profile._times_could_3bet = row[6]
            profile._times_did_3bet = row[7]
            profile._times_faced_3bet = row[8]
            profile._times_folded_to_3bet = row[9]
            profile._times_could_cbet = row[10]
            profile._times_did_cbet = row[11]
            profile._times_faced_cbet = row[12]
            profile._times_folded_to_cbet = row[13]
            profile._total_bets_raises = row[14]
            profile._total_calls = row[15]
            profile._times_reached_showdown = row[16]
            profile._times_won_showdown = row[17]
            profile._times_could_check_raise = row[18]
            profile._times_did_check_raise = row[19]
            profile.recalculate()

        self._cache[name] = profile
        return profile

    def save_profile(self, profile: PlayerProfile):
        """Save a player profile to the database."""
        self.conn.execute("""
            INSERT OR REPLACE INTO players VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            profile.name, profile.hands_played,
            profile._times_could_vpip, profile._times_did_vpip,
            profile._times_could_pfr, profile._times_did_pfr,
            profile._times_could_3bet, profile._times_did_3bet,
            profile._times_faced_3bet, profile._times_folded_to_3bet,
            profile._times_could_cbet, profile._times_did_cbet,
            profile._times_faced_cbet, profile._times_folded_to_cbet,
            profile._total_bets_raises, profile._total_calls,
            profile._times_reached_showdown, profile._times_won_showdown,
            profile._times_could_check_raise, profile._times_did_check_raise,
        ))
        self.conn.commit()
        self._cache[profile.name] = profile

    def record_hand(self, player_name: str, actions: list, went_to_showdown: bool,
                    won_at_showdown: bool, was_preflop_raiser: bool):
        """Record a completed hand for a player and update their stats."""
        profile = self.get_profile(player_name)
        profile.hands_played += 1

        # VPIP — did they voluntarily put money in?
        profile._times_could_vpip += 1
        voluntary_actions = {"call", "bet", "raise", "all_in"}
        preflop_actions = [a for a in actions if a.get("street") == "preflop"]
        if any(a["action"] in voluntary_actions for a in preflop_actions):
            profile._times_did_vpip += 1

        # PFR — did they raise preflop?
        profile._times_could_pfr += 1
        if any(a["action"] in ("raise", "all_in") for a in preflop_actions):
            profile._times_did_pfr += 1

        # Count bets/raises vs calls for AF
        for a in actions:
            if a["action"] in ("bet", "raise", "all_in"):
                profile._total_bets_raises += 1
            elif a["action"] == "call":
                profile._total_calls += 1

        # Showdown
        if went_to_showdown:
            profile._times_reached_showdown += 1
            if won_at_showdown:
                profile._times_won_showdown += 1

        profile.recalculate()
        self.save_profile(profile)

    def save_all(self):
        """Save all cached profiles to database."""
        for profile in self._cache.values():
            self.save_profile(profile)

    def close(self):
        self.save_all()
        self.conn.close()

    def __del__(self):
        """Ensure database connection is closed on garbage collection."""
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
