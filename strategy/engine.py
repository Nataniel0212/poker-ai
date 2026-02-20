"""
Strategy engine — combines equity calculation, pre-flop charts, GTO lookup,
and exploit adjustments to produce optimal recommendations.
"""

import random
from typing import Optional
from dataclasses import dataclass
from strategy.push_fold import should_push, should_call_allin, get_push_range


@dataclass
class Recommendation:
    action: str             # "fold", "check", "call", "bet", "raise"
    amount: float = 0.0     # Bet/raise amount
    confidence: float = 0.0 # 0-100
    equity: float = 0.0     # Hand equity vs estimated range
    pot_odds: float = 0.0
    ev: float = 0.0         # Expected value
    reasoning: str = ""
    if_raised: str = ""     # What to do if opponent raises
    if_called: str = ""     # What to do if opponent calls (plan for next street)
    exploit_note: str = ""  # Specific exploit being used


# ========== PRE-FLOP CHARTS (GTO-based, 6-max) ==========

# Hands represented as sorted rank pairs: 'AA', 'AKs', 'AKo', etc.
# s = suited, o = offsuit, no suffix = pair

# RFI (Raise First In) ranges by position — standard 6-max GTO
OPEN_RAISE_RANGES = {
    "UTG": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AQs", "AJs", "ATs", "A5s", "A4s",
        "KQs", "KJs", "KTs",
        "QJs", "QTs",
        "JTs",
        "T9s",
        "98s",
        "87s",
        "AKo", "AQo",
    },
    "HJ": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A5s", "A4s", "A3s",
        "KQs", "KJs", "KTs", "K9s",
        "QJs", "QTs", "Q9s",
        "JTs", "J9s",
        "T9s", "T8s",
        "98s", "97s",
        "87s", "86s",
        "76s",
        "65s",
        "AKo", "AQo", "AJo",
        "KQo",
    },
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "QJs", "QTs", "Q9s", "Q8s",
        "JTs", "J9s", "J8s",
        "T9s", "T8s",
        "98s", "97s",
        "87s", "86s",
        "76s", "75s",
        "65s", "64s",
        "54s",
        "AKo", "AQo", "AJo", "ATo",
        "KQo", "KJo",
        "QJo",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s",
        "JTs", "J9s", "J8s", "J7s",
        "T9s", "T8s", "T7s",
        "98s", "97s", "96s",
        "87s", "86s", "85s",
        "76s", "75s",
        "65s", "64s",
        "54s", "53s",
        "43s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o",
        "KQo", "KJo", "KTo",
        "QJo", "QTo",
        "JTo",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s",
        "QJs", "QTs", "Q9s", "Q8s",
        "JTs", "J9s", "J8s",
        "T9s", "T8s",
        "98s", "97s",
        "87s", "86s",
        "76s", "75s",
        "65s",
        "54s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQo", "KJo",
        "QJo",
    },
    "BB": {
        # BB defends wide vs steals — but for RFI this is rarely used
        # (BB typically checks or defends, not opens)
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "QJs", "QTs", "Q9s",
        "JTs", "J9s",
        "T9s", "T8s",
        "98s", "97s",
        "87s", "86s",
        "76s",
        "65s",
        "54s",
        "AKo", "AQo", "AJo", "ATo",
        "KQo", "KJo",
        "QJo",
    },
}

# 3-bet ranges (vs open raise)
THREE_BET_RANGES = {
    "value": {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"},
    "bluff": {"A5s", "A4s", "A3s", "A2s", "76s", "65s", "54s"},
}


def hand_to_notation(card1: str, card2: str) -> str:
    """Convert two cards (e.g. 'As', 'Kh') to notation like 'AKs' or 'AKo'.
    Handles '10x' format by converting to 'Tx'."""
    rank_order = "AKQJT98765432"

    # Normalize "10" to "T"
    if card1.startswith("10"):
        card1 = "T" + card1[2:]
    if card2.startswith("10"):
        card2 = "T" + card2[2:]

    r1 = card1[0].upper()
    r2 = card2[0].upper()
    s1 = card1[1].lower()
    s2 = card2[1].lower()

    # Sort by rank
    if rank_order.index(r1) > rank_order.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1

    if r1 == r2:
        return f"{r1}{r2}"  # Pair
    elif s1 == s2:
        return f"{r1}{r2}s"  # Suited
    else:
        return f"{r1}{r2}o"  # Offsuit


class EquityCalculator:
    """Hand equity calculator using Monte Carlo simulation."""

    def __init__(self):
        try:
            from treys import Card, Evaluator
            self.evaluator = Evaluator()
            self.Card = Card
            self._use_treys = True
        except ImportError:
            self._use_treys = False
            print("Warning: treys not installed, equity calculations will be estimated")

    def calculate_equity(self, hero_cards: list, community_cards: list,
                         num_opponents: int = 1, num_simulations: int = 10000) -> float:
        """Calculate hand equity via Monte Carlo simulation.

        Args:
            hero_cards: ['As', 'Kh'] — hero's hole cards
            community_cards: ['Js', 'Ts', '3d'] — board cards
            num_opponents: number of opponents
            num_simulations: number of Monte Carlo iterations

        Returns:
            Equity as a float between 0 and 1
        """
        if not self._use_treys:
            return self._estimate_equity(hero_cards, community_cards, num_opponents)

        Card = self.Card

        def parse_card(c: str) -> int:
            """Convert 'As' or '10s' to treys Card int."""
            # Normalize "10" to "T"
            if c.startswith("10"):
                c = "T" + c[2:]
            rank = c[0].upper()
            suit = c[1].lower()
            return Card.new(f"{rank}{suit}")

        try:
            hero = [parse_card(c) for c in hero_cards]
            board = [parse_card(c) for c in community_cards]
        except Exception:
            return 0.5

        # Validate: no duplicate cards (OCR sometimes misreads)
        all_known = hero + board
        if len(set(all_known)) != len(all_known):
            return self._estimate_equity(hero_cards, community_cards, num_opponents)

        # All cards in a standard deck
        full_deck = [Card.new(f"{r}{s}") for r in "AKQJT98765432" for s in "shdc"]
        used = set(hero + board)
        remaining = [c for c in full_deck if c not in used]

        wins = 0
        ties = 0

        for _ in range(num_simulations):
            random.shuffle(remaining)
            idx = 0

            # Deal cards to opponents
            opp_hands = []
            for _ in range(num_opponents):
                opp_hands.append([remaining[idx], remaining[idx + 1]])
                idx += 2

            # Complete the board
            sim_board = board[:]
            while len(sim_board) < 5:
                sim_board.append(remaining[idx])
                idx += 1

            # Evaluate
            hero_score = self.evaluator.evaluate(sim_board, hero)
            best_opp = min(
                self.evaluator.evaluate(sim_board, opp) for opp in opp_hands
            )

            if hero_score < best_opp:  # Lower is better in treys
                wins += 1
            elif hero_score == best_opp:
                ties += 1

        if num_simulations == 0:
            return 0.5
        return (wins + ties * 0.5) / num_simulations

    def _estimate_equity(self, hero_cards, community_cards, num_opponents):
        """Rough equity estimate when treys is not available."""
        if not hero_cards or len(hero_cards) < 2:
            return 0.5
        # Simple heuristic based on hand strength
        notation = hand_to_notation(hero_cards[0], hero_cards[1])
        premium = {"AA", "KK", "QQ", "JJ", "AKs", "AKo"}
        strong = {"TT", "99", "AQs", "AQo", "AJs", "KQs"}
        medium = {"88", "77", "ATs", "AJo", "KJs", "QJs", "JTs"}

        if notation in premium:
            base = 0.75
        elif notation in strong:
            base = 0.60
        elif notation in medium:
            base = 0.50
        else:
            base = 0.40

        # Adjust for community cards (postflop hands are more defined)
        if community_cards:
            # Check for basic flush/straight potential
            hero_ranks = [c[0].upper() for c in hero_cards]
            hero_suits = [c[-1].lower() for c in hero_cards]
            board_ranks = [c[0].upper() for c in community_cards]
            board_suits = [c[-1].lower() for c in community_cards]

            # Pair on board boost
            all_ranks = hero_ranks + board_ranks
            for r in hero_ranks:
                if all_ranks.count(r) >= 2:
                    base += 0.10  # Paired hand
                    break

            # Flush draw boost
            for s in hero_suits:
                suited_count = board_suits.count(s) + hero_suits.count(s)
                if suited_count >= 4:
                    base += 0.08  # Flush draw
                    break

            # Reduce uncertainty as more cards are revealed
            if len(community_cards) >= 4:
                base = base * 0.9 + 0.1 * (1 if base > 0.5 else 0)

        # Adjust for number of opponents
        base *= (1 - 0.05 * (num_opponents - 1))
        return max(0.1, min(0.95, base))


class StrategyEngine:
    """Main strategy engine — combines all analysis into recommendations."""

    def __init__(self):
        self.equity_calc = EquityCalculator()

    def analyze(self, game_context: dict, opponent_profile=None) -> Recommendation:
        """Produce a recommendation based on current game state and opponent data.

        Args:
            game_context: dict from GameState.get_context_summary()
            opponent_profile: PlayerProfile of main villain (optional)
        """
        street = game_context.get("street", "preflop")

        if street == "preflop":
            return self._analyze_preflop(game_context, opponent_profile)
        else:
            return self._analyze_postflop(game_context, opponent_profile)

    def _analyze_preflop(self, ctx: dict, villain=None) -> Recommendation:
        """Pre-flop decision making."""
        hero_cards = ctx.get("hero_cards")
        if not hero_cards or len(hero_cards) != 2:
            return Recommendation(action="fold", reasoning="Kan inte se korten")

        position = ctx.get("hero_position", "UTG")
        bb = ctx.get("big_blind", 1.0)
        if bb <= 0:
            bb = 1.0
        hero_stack = ctx.get("hero_stack", 100)
        stack_bb = hero_stack / bb
        notation = hand_to_notation(hero_cards[0], hero_cards[1])

        actions = ctx.get("actions_this_street", [])
        facing_raise = any(a["action"] in ("raise", "all_in") for a in actions)
        facing_3bet = sum(1 for a in actions if a["action"] == "raise") >= 2
        facing_allin = any(a["action"] == "all_in" for a in actions)

        # Short stack push/fold mode (< 15BB in tournaments)
        is_tournament = ctx.get("is_tournament", False)
        if stack_bb <= 15 and is_tournament and not facing_allin:
            if should_push(notation, position, stack_bb):
                rec = Recommendation(
                    action="raise",
                    amount=hero_stack,  # All-in
                    confidence=80,
                    reasoning=f"Push/fold: {notation} ar en push med {stack_bb:.0f}BB fran {position}",
                    if_raised="Vi ar redan all-in",
                )
                rec.equity = self.equity_calc.calculate_equity(
                    hero_cards, [], max(1, ctx.get("num_active_players", 2) - 1), 5000
                )
                return rec
            else:
                return Recommendation(
                    action="fold",
                    confidence=80,
                    reasoning=f"Push/fold: {notation} ar en fold med {stack_bb:.0f}BB fran {position}",
                )

        # Facing all-in
        if facing_allin:
            pusher_pos = "BTN"  # Default
            for a in actions:
                if a["action"] == "all_in":
                    # Try to find pusher position from villains
                    for v in ctx.get("villains", []):
                        if v.get("name") == a.get("player"):
                            pusher_pos = v.get("position", "BTN")
                            break

            if should_call_allin(notation, position, pusher_pos):
                return Recommendation(
                    action="call",
                    confidence=75,
                    reasoning=f"Call all-in med {notation} vs {pusher_pos} push range",
                    equity=self.equity_calc.calculate_equity(
                        hero_cards, [], 1, 5000
                    ),
                )
            else:
                return Recommendation(
                    action="fold",
                    confidence=75,
                    reasoning=f"Fold {notation} vs all-in fran {pusher_pos}",
                )

        # Check if hand is in our opening range
        open_range = OPEN_RAISE_RANGES.get(position, OPEN_RAISE_RANGES["UTG"])
        in_range = notation in open_range

        if facing_3bet:
            # Facing 3-bet: only continue with value hands
            if notation in THREE_BET_RANGES["value"]:
                rec = Recommendation(
                    action="raise",
                    amount=ctx.get("pot", bb * 3) * 2.5,
                    confidence=85,
                    reasoning=f"4-bet med {notation} — premiumhand vs 3-bet",
                )
            elif notation in {"TT", "99", "AQs", "AJs", "KQs"}:
                rec = Recommendation(
                    action="call",
                    confidence=60,
                    reasoning=f"Call 3-bet med {notation} — bra implied odds",
                )
            else:
                rec = Recommendation(
                    action="fold",
                    confidence=80,
                    reasoning=f"Fold {notation} vs 3-bet — utanför continue-range",
                )
        elif facing_raise:
            # Facing open raise
            if notation in THREE_BET_RANGES["value"]:
                rec = Recommendation(
                    action="raise",
                    amount=ctx.get("pot", bb * 3) * 3,
                    confidence=85,
                    reasoning=f"3-bet value med {notation}",
                )
            elif notation in THREE_BET_RANGES["bluff"]:
                rec = Recommendation(
                    action="raise",
                    amount=ctx.get("pot", bb * 3) * 3,
                    confidence=55,
                    reasoning=f"3-bet bluff med {notation} — bra blocker",
                )
            elif in_range:
                rec = Recommendation(
                    action="call",
                    confidence=65,
                    reasoning=f"Call open med {notation} — i range för {position}",
                )
            else:
                rec = Recommendation(
                    action="fold",
                    confidence=80,
                    reasoning=f"Fold {notation} vs raise — utanför call-range",
                )
        else:
            # First to act / limpers only
            if in_range:
                raise_size = bb * 2.5
                if position in ("SB", "BB"):
                    raise_size = bb * 3
                rec = Recommendation(
                    action="raise",
                    amount=raise_size,
                    confidence=75,
                    reasoning=f"Open raise {notation} från {position}",
                )
            else:
                rec = Recommendation(
                    action="fold",
                    confidence=85,
                    reasoning=f"{notation} utanför open-range för {position}",
                )

        # Apply exploit adjustments
        if villain and villain.hands_played >= 20:
            rec = self._apply_preflop_exploits(rec, ctx, villain, notation)

        # Calculate equity
        if hero_cards:
            num_opp = max(1, ctx.get("num_active_players", 2) - 1)
            rec.equity = self.equity_calc.calculate_equity(
                hero_cards, [], num_opp, num_simulations=5000
            )

        return rec

    def _apply_preflop_exploits(self, rec: Recommendation, ctx: dict,
                                villain, notation: str) -> Recommendation:
        """Adjust preflop recommendation based on opponent tendencies."""
        # If villain folds to 3-bet a lot, 3-bet lighter
        if villain.fold_to_three_bet > 65 and rec.action == "call":
            open_range = OPEN_RAISE_RANGES.get("BTN", set())
            if notation in open_range:
                rec.action = "raise"
                rec.amount = ctx.get("pot", 3) * 3
                rec.confidence = 65
                rec.exploit_note = f"Exploit: 3-bet light — villain foldar {villain.fold_to_three_bet:.0f}% till 3-bet"

        # If villain is very tight (nit), steal more
        if villain.vpip < 18 and rec.action == "fold":
            position = ctx.get("hero_position", "")
            if position in ("BTN", "CO", "SB"):
                wider_range = OPEN_RAISE_RANGES.get("BTN", set())
                if notation in wider_range:
                    rec.action = "raise"
                    rec.amount = ctx.get("big_blind", 1) * 2.5
                    rec.confidence = 60
                    rec.exploit_note = f"Exploit: steal vs nit (VPIP {villain.vpip:.0f}%)"

        return rec

    def _analyze_postflop(self, ctx: dict, villain=None) -> Recommendation:
        """Post-flop decision making."""
        hero_cards = ctx.get("hero_cards")
        community = ctx.get("community_cards", [])
        pot = ctx.get("pot", 0)
        bb = ctx.get("big_blind", 1)

        if not hero_cards:
            return Recommendation(action="check", reasoning="Kan inte se korten")

        # Calculate equity
        num_opp = max(1, ctx.get("num_active_players", 2) - 1)
        equity = self.equity_calc.calculate_equity(
            hero_cards, community, num_opp, num_simulations=10000
        )

        # Check what actions we're facing
        actions = ctx.get("actions_this_street", [])
        villain_names = {v.get("name") for v in ctx.get("villains", [])}
        facing_bet = any(
            a["action"] in ("bet", "raise") for a in actions
            if a["player"] in villain_names
        )

        # Calculate pot odds if facing a bet
        pot_odds = 0
        bet_to_call = 0
        if facing_bet and actions:
            last_bet = next(
                (a for a in reversed(actions) if a["action"] in ("bet", "raise")),
                None,
            )
            if last_bet:
                bet_to_call = last_bet["amount"]
                pot_odds = bet_to_call / (pot + bet_to_call) if (pot + bet_to_call) > 0 else 0

        # Decision logic
        if facing_bet:
            ev = equity * (pot + bet_to_call) - (1 - equity) * bet_to_call

            if equity > 0.65:
                # Strong hand — raise for value
                raise_amount = pot * 0.75
                rec = Recommendation(
                    action="raise",
                    amount=raise_amount,
                    confidence=int(equity * 100),
                    equity=equity,
                    pot_odds=pot_odds,
                    ev=ev,
                    reasoning=f"Raise for value — equity {equity:.0%} vs range",
                    if_raised="Call om vi har odds, annars fold",
                    if_called=f"Fortsätt value bet om equity håller",
                )
            elif equity > pot_odds and equity > 0.35:
                # Profitable call
                rec = Recommendation(
                    action="call",
                    confidence=int(min(80, equity * 100)),
                    equity=equity,
                    pot_odds=pot_odds,
                    ev=ev,
                    reasoning=f"Call — equity {equity:.0%} > pot odds {pot_odds:.0%}",
                    if_raised="Fold — vi callar marginellt",
                )
            else:
                rec = Recommendation(
                    action="fold",
                    confidence=int((1 - equity) * 80),
                    equity=equity,
                    pot_odds=pot_odds,
                    ev=ev,
                    reasoning=f"Fold — equity {equity:.0%} < pot odds {pot_odds:.0%}",
                )
        else:
            # We have option to bet or check
            if equity > 0.60:
                # Value bet
                bet_size = pot * 0.66
                rec = Recommendation(
                    action="bet",
                    amount=bet_size,
                    confidence=int(equity * 100),
                    equity=equity,
                    ev=equity * pot,
                    reasoning=f"Value bet — equity {equity:.0%}, bet 2/3 pot",
                    if_raised="Call om equity > 50%, annars fold",
                    if_called="Fortsätt value bet nästa street om equity håller",
                )
            elif equity > 0.40:
                # Medium strength — check for pot control
                rec = Recommendation(
                    action="check",
                    confidence=55,
                    equity=equity,
                    ev=0,
                    reasoning=f"Check — medium styrka ({equity:.0%}), pot control",
                    if_raised="Call en liten bet, fold stora bets",
                )
            elif equity > 0.20:
                # Weak hand but might have fold equity
                bet_size = pot * 0.5
                fold_equity_needed = bet_size / (pot + bet_size)
                rec = Recommendation(
                    action="bet",
                    amount=bet_size,
                    confidence=40,
                    equity=equity,
                    reasoning=f"Semi-bluff — equity {equity:.0%}, behöver {fold_equity_needed:.0%} fold equity",
                    if_raised="Fold — vår bluff blev caught",
                )
            else:
                rec = Recommendation(
                    action="check",
                    confidence=70,
                    equity=equity,
                    reasoning=f"Check/fold — svag hand ({equity:.0%})",
                    if_raised="Fold",
                )

        # Apply exploit adjustments
        if villain and villain.hands_played >= 20:
            rec = self._apply_postflop_exploits(rec, ctx, villain, equity)

        return rec

    def _apply_postflop_exploits(self, rec: Recommendation, ctx: dict,
                                  villain, equity: float) -> Recommendation:
        """Adjust postflop recommendation based on opponent tendencies."""
        pot = ctx.get("pot", 0)

        # Exploit: opponent folds to c-bets too much
        if villain.fold_to_cbet > 65 and rec.action == "check" and equity > 0.15:
            rec.action = "bet"
            rec.amount = pot * 0.5
            rec.confidence = max(rec.confidence, 60)
            rec.exploit_note = f"Exploit: c-bet bluff — villain foldar {villain.fold_to_cbet:.0f}% till c-bet"

        # Exploit: opponent is a calling station — never bluff, value bet thin
        if villain.player_type == "calling_station":
            if rec.action == "bet" and equity < 0.40:
                rec.action = "check"
                rec.confidence = 65
                rec.exploit_note = "Exploit: check vs calling station — bluffar aldrig"
            elif rec.action == "bet" and equity > 0.50:
                rec.amount = pot * 0.85  # Bet bigger for value
                rec.exploit_note = "Exploit: bigger value bet vs calling station"

        # Exploit: opponent is a maniac — trap more
        if villain.player_type == "maniac" and equity > 0.60 and rec.action == "bet":
            rec.action = "check"
            rec.confidence = 70
            rec.exploit_note = "Exploit: trap vs maniac — låt dem bluffa"

        # Exploit: opponent rarely goes to showdown — bluff rivers more
        if villain.wtsd < 22 and rec.action == "check" and equity < 0.35:
            street = ctx.get("street", "")
            if street == "river":
                rec.action = "bet"
                rec.amount = pot * 0.75
                rec.confidence = 55
                rec.exploit_note = f"Exploit: river bluff — villain går bara till SD {villain.wtsd:.0f}%"

        return rec

    def get_pot_odds(self, pot: float, bet_to_call: float) -> float:
        """Calculate pot odds as a ratio."""
        if pot + bet_to_call == 0:
            return 0
        return bet_to_call / (pot + bet_to_call)

    def calculate_icm(self, stacks: list, payouts: list) -> list:
        """Calculate ICM equity for tournament play.

        Args:
            stacks: list of chip stacks for each player
            payouts: list of payouts [1st, 2nd, 3rd, ...]

        Returns:
            list of ICM equity (dollar values) for each player
        """
        n = len(stacks)
        total = sum(stacks)
        if total == 0:
            return [0.0] * n

        # Cap recursion depth to avoid factorial explosion with many players
        # For >10 players, use simplified probability-based approximation
        max_payout_depth = min(len(payouts), 5)  # Only recurse for top 5 places
        if n > 10:
            # Simplified ICM: equity proportional to stack with payout weighting
            probabilities = [s / total for s in stacks]
            total_payout = sum(payouts)
            return [p * total_payout for p in probabilities]

        equities = [0.0] * n

        def _icm_recursive(remaining: list, place: int, prob_product: float):
            if place >= max_payout_depth or not remaining:
                return

            remaining_total = sum(stacks[j] for j in remaining)
            if remaining_total == 0:
                return

            for i in remaining:
                p = stacks[i] / remaining_total
                equity_gain = p * prob_product * payouts[place]
                equities[i] += equity_gain

                new_remaining = [j for j in remaining if j != i]
                _icm_recursive(new_remaining, place + 1, prob_product * p)

        _icm_recursive(list(range(n)), 0, 1.0)
        return equities
