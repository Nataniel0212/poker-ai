"""
HH-based Opponent Tracker.

Takes parsed HHHand objects from hh_parser and calculates per-player stats:
VPIP, PFR, AF, 3-bet%, fold-to-3bet, c-bet%, fold-to-cbet, WTSD, W$SD.

Updates the OpponentDatabase with exact stats derived from hand history,
which is far more reliable than OCR-inferred actions.
"""

from history.hh_parser import HHHand, HHAction
from profiles.opponent_db import OpponentDatabase


class OpponentTracker:
    """Processes parsed hands and updates opponent profiles."""

    def __init__(self, opponent_db: OpponentDatabase, hero_name: str = ""):
        self.db = opponent_db
        self.hero_name = hero_name
        self.hands_processed = 0

    def process_hand(self, hand: HHHand):
        """Process a single parsed hand and update all opponent profiles.

        This replaces the old _record_completed_hand() logic in main.py
        with much more accurate data from hand history files.
        """
        if not hand.players:
            return

        # Determine hero name from the hand if not set
        hero = self.hero_name or hand.hero_name

        # Analyze preflop action sequence
        preflop_non_blind = [a for a in hand.preflop_actions
                             if a.action != "post_blind"]

        # Find open raiser (first voluntary raiser preflop)
        open_raiser = None
        three_bettor = None
        raise_count = 0
        for a in preflop_non_blind:
            if a.action in ("raise", "all_in"):
                raise_count += 1
                if raise_count == 1:
                    open_raiser = a.player
                elif raise_count == 2:
                    three_bettor = a.player
                    break

        # Find who could have 3-bet (acted after the open raiser)
        could_3bet_players = set()
        if open_raiser:
            seen_open = False
            for a in preflop_non_blind:
                if a.player == open_raiser and a.action in ("raise", "all_in"):
                    seen_open = True
                    continue
                if seen_open and a.player != open_raiser:
                    could_3bet_players.add(a.player)

        # Analyze flop for c-bet detection
        flop_bettor = None
        for a in hand.flop_actions:
            if a.action in ("bet", "raise", "all_in"):
                flop_bettor = a.player
                break

        # Determine who faced the c-bet on flop (acted after the c-bettor)
        faced_cbet_players = set()
        folded_to_cbet_players = set()
        if flop_bettor and flop_bettor == open_raiser:
            seen_cbet = False
            for a in hand.flop_actions:
                if a.player == flop_bettor and a.action in ("bet", "raise", "all_in"):
                    seen_cbet = True
                    continue
                if seen_cbet and a.player != flop_bettor:
                    faced_cbet_players.add(a.player)
                    if a.action == "fold":
                        folded_to_cbet_players.add(a.player)

        # Determine who faced a 3-bet (the open raiser, if someone 3-bet)
        faced_3bet_players = set()
        folded_to_3bet_players = set()
        if open_raiser and three_bettor:
            faced_3bet_players.add(open_raiser)
            # Check if open raiser folded preflop after the 3-bet
            seen_3bet = False
            for a in preflop_non_blind:
                if a.player == three_bettor and a.action in ("raise", "all_in"):
                    seen_3bet = True
                    continue
                if seen_3bet and a.player == open_raiser:
                    if a.action == "fold":
                        folded_to_3bet_players.add(open_raiser)
                    break

        # Process each non-hero player
        for player in hand.players:
            if player.name == hero:
                continue

            name = player.name
            player_actions = hand.player_actions(name)
            if not player_actions:
                continue

            # Build action dicts for record_hand (compatible with existing API)
            action_dicts = [
                {
                    "action": a.action,
                    "street": a.street,
                    "amount": a.amount,
                }
                for a in player_actions
                if a.action != "post_blind"
            ]

            preflop_player_actions = [a for a in player_actions
                                      if a.street == "preflop"
                                      and a.action != "post_blind"]

            was_pfr = any(a.action in ("raise", "all_in")
                          for a in preflop_player_actions)

            went_to_sd = hand.player_went_to_showdown(name)
            won_at_sd = hand.player_won_at_showdown(name)

            # 3-bet stats
            could_3bet = name in could_3bet_players
            did_3bet = (three_bettor == name)
            faced_3bet = name in faced_3bet_players
            folded_to_3bet = name in folded_to_3bet_players

            # C-bet stats
            could_cbet = (open_raiser == name and len(hand.flop) >= 3)
            did_cbet = (could_cbet and flop_bettor == name)
            faced_cbet = name in faced_cbet_players
            folded_to_cbet = name in folded_to_cbet_players

            self.db.record_hand(
                player_name=name,
                actions=action_dicts,
                went_to_showdown=went_to_sd,
                won_at_showdown=won_at_sd,
                was_preflop_raiser=was_pfr,
                faced_3bet=faced_3bet,
                folded_to_3bet=folded_to_3bet,
                could_3bet=could_3bet,
                did_3bet=did_3bet,
                could_cbet=could_cbet,
                did_cbet=did_cbet,
                faced_cbet=faced_cbet,
                folded_to_cbet=folded_to_cbet,
            )

        self.hands_processed += 1

    def process_hands(self, hands: list):
        """Process multiple hands (e.g., when loading existing history)."""
        for hand in hands:
            self.process_hand(hand)
        if hands:
            print(f"Opponent Tracker: Processed {len(hands)} hands, "
                  f"total {self.hands_processed}")

    def get_stats_summary(self, player_name: str) -> str:
        """Get a formatted stats summary for a player."""
        profile = self.db.get_profile(player_name)
        return profile.summary_string()
