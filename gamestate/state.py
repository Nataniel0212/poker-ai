"""
Game state tracking — holds all information about the current hand,
players, actions, and tournament status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Street(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_BLIND = "post_blind"


class Position(Enum):
    BTN = "BTN"   # Button / Dealer
    SB = "SB"     # Small Blind
    BB = "BB"     # Big Blind
    UTG = "UTG"   # Under The Gun
    UTG1 = "UTG+1"
    UTG2 = "UTG+2"
    MP = "MP"     # Middle Position
    MP1 = "MP+1"
    HJ = "HJ"    # Hijack
    CO = "CO"     # Cutoff


# Map number of players to position names
POSITION_MAP = {
    2: [Position.SB, Position.BB],
    3: [Position.BTN, Position.SB, Position.BB],
    4: [Position.BTN, Position.SB, Position.BB, Position.UTG],
    5: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.CO],
    6: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.HJ, Position.CO],
    7: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.UTG1, Position.HJ, Position.CO],
    8: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.UTG1, Position.MP, Position.HJ, Position.CO],
    9: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.UTG1, Position.UTG2, Position.MP, Position.HJ, Position.CO],
}


@dataclass
class Action:
    player_name: str
    action_type: ActionType
    amount: float = 0.0
    street: Street = Street.PREFLOP
    timestamp: float = field(default_factory=time.time)


@dataclass
class Player:
    name: str
    seat: int
    stack: float
    position: Optional[Position] = None
    hole_cards: Optional[list] = None  # e.g. ['As', 'Kh']
    is_active: bool = True  # Still in the hand
    is_hero: bool = False   # Is this us?
    current_bet: float = 0.0

    def reset_for_new_hand(self):
        self.hole_cards = None
        self.is_active = True
        self.current_bet = 0.0


@dataclass
class Hand:
    hand_number: int = 0
    players: list = field(default_factory=list)
    community_cards: list = field(default_factory=list)  # e.g. ['Js', '10s', '3d']
    pot: float = 0.0
    side_pots: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    street: Street = Street.PREFLOP
    dealer_seat: int = 0
    small_blind: float = 0.0
    big_blind: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # Tournament specific
    is_tournament: bool = False
    tournament_level: int = 0
    players_remaining: int = 0
    total_players: int = 0
    payout_structure: dict = field(default_factory=dict)

    def get_hero(self) -> Optional[Player]:
        for p in self.players:
            if p.is_hero:
                return p
        return None

    def get_active_players(self) -> list:
        return [p for p in self.players if p.is_active]

    def get_player_by_name(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    def get_actions_on_street(self, street: Street) -> list:
        return [a for a in self.actions if a.street == street]

    def get_actions_by_player(self, name: str) -> list:
        return [a for a in self.actions if a.player_name == name]

    def get_villain(self) -> Optional[Player]:
        """Get the main opponent (in heads-up pots). Returns None if multiway."""
        active = [p for p in self.get_active_players() if not p.is_hero]
        if len(active) == 1:
            return active[0]
        return None

    def get_villains(self) -> list:
        """Get all active opponents."""
        return [p for p in self.get_active_players() if not p.is_hero]


class GameState:
    """Main game state manager — tracks current hand and session history."""

    def __init__(self):
        self.current_hand: Optional[Hand] = None
        self.hand_history: list = []  # List of completed Hand objects
        self.session_profit: float = 0.0
        self.hands_played: int = 0

    def new_hand(self, hand_number: int, players: list, dealer_seat: int,
                 small_blind: float, big_blind: float):
        """Start a new hand."""
        if self.current_hand:
            self.hand_history.append(self.current_hand)

        self.current_hand = Hand(
            hand_number=hand_number,
            players=players,
            dealer_seat=dealer_seat,
            small_blind=small_blind,
            big_blind=big_blind,
        )
        self.hands_played += 1

        # Assign positions based on dealer and number of players
        self._assign_positions()

    def _assign_positions(self):
        """Assign positions relative to the dealer button."""
        hand = self.current_hand
        if not hand:
            return

        n_players = len(hand.players)
        if n_players not in POSITION_MAP or n_players == 0:
            return

        positions = POSITION_MAP[n_players]

        # Sort players by seat, starting from one after dealer
        seats = sorted([p.seat for p in hand.players])
        if not seats:
            return
        dealer_idx = seats.index(hand.dealer_seat) if hand.dealer_seat in seats else 0

        for i, pos in enumerate(positions):
            seat_idx = (dealer_idx + i) % n_players
            seat = seats[seat_idx]
            player = next(p for p in hand.players if p.seat == seat)
            player.position = pos

    def add_action(self, player_name: str, action_type: ActionType, amount: float = 0.0):
        """Record a player action."""
        if not self.current_hand:
            return

        action = Action(
            player_name=player_name,
            action_type=action_type,
            amount=amount,
            street=self.current_hand.street,
        )
        self.current_hand.actions.append(action)

        # Update player state
        player = self.current_hand.get_player_by_name(player_name)
        if player:
            if action_type == ActionType.FOLD:
                player.is_active = False
            elif action_type in (ActionType.BET, ActionType.RAISE, ActionType.CALL, ActionType.ALL_IN):
                player.current_bet += amount

    def update_community_cards(self, cards: list):
        """Update community cards and advance street."""
        if not self.current_hand:
            return

        old_street = self.current_hand.street
        self.current_hand.community_cards = cards

        new_street = old_street
        if len(cards) == 3:
            new_street = Street.FLOP
        elif len(cards) == 4:
            new_street = Street.TURN
        elif len(cards) == 5:
            new_street = Street.RIVER

        # Only reset bets when street actually changes
        if new_street != old_street:
            self.current_hand.street = new_street
            for player in self.current_hand.players:
                player.current_bet = 0.0

    def update_pot(self, pot: float):
        if self.current_hand:
            self.current_hand.pot = pot

    def set_hero_cards(self, cards: list):
        if self.current_hand:
            hero = self.current_hand.get_hero()
            if hero:
                hero.hole_cards = cards

    def get_context_summary(self) -> dict:
        """Get a summary of the current game state for the strategy engine."""
        hand = self.current_hand
        if not hand:
            return {}

        hero = hand.get_hero()
        villains = hand.get_villains()

        return {
            "hand_number": hand.hand_number,
            "hero_cards": hero.hole_cards if hero else None,
            "hero_position": hero.position.value if hero and hero.position else None,
            "hero_stack": hero.stack if hero else 0,
            "community_cards": hand.community_cards,
            "pot": hand.pot,
            "street": hand.street.value,
            "big_blind": hand.big_blind,
            "num_active_players": len(hand.get_active_players()),
            "villains": [
                {
                    "name": v.name,
                    "position": v.position.value if v.position else None,
                    "stack": v.stack,
                    "current_bet": v.current_bet,
                }
                for v in villains
            ],
            "actions_this_street": [
                {
                    "player": a.player_name,
                    "action": a.action_type.value,
                    "amount": a.amount,
                }
                for a in hand.get_actions_on_street(hand.street)
            ],
            "all_actions": [
                {
                    "player": a.player_name,
                    "action": a.action_type.value,
                    "amount": a.amount,
                    "street": a.street.value,
                }
                for a in hand.actions
            ],
            "is_tournament": hand.is_tournament,
            "tournament_level": hand.tournament_level,
            "players_remaining": hand.players_remaining,
        }
