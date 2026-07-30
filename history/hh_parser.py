"""
PokerStars Hand History Parser.

Monitors a hand history folder for new/updated files, parses each completed
hand into structured data. Supports cash games (No Limit Hold'em).

PokerStars HH format example:
    PokerStars Hand #242850173477: Hold'em No Limit ($0.25/$0.50 USD) - 2026/02/20 14:33:05 CET
    Table 'Lalage IV' 6-max Seat #3 is the button
    Seat 1: Villain1 ($52.35 in chips)
    Seat 3: Hero ($50.00 in chips)
    Seat 5: Villain2 ($48.75 in chips)
    Villain2: posts small blind $0.25
    Villain1: posts big blind $0.50
    *** HOLE CARDS ***
    Dealt to Hero [As Kh]
    Hero: raises $1.00 to $1.50
    Villain2: folds
    Villain1: calls $1.00
    *** FLOP *** [Js Ts 3d]
    Villain1: checks
    Hero: bets $2.25
    Villain1: folds
    Uncalled bet ($2.25) returned to Hero
    Hero collected $3.25 from pot
    *** SUMMARY ***
    Total pot $3.25 | Rake $0.15
    Board [Js Ts 3d]
"""

import re
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class HHPlayer:
    """A player in a parsed hand."""
    name: str
    seat: int
    stack: float
    hole_cards: Optional[list] = None  # ['As', 'Kh'] if shown
    is_hero: bool = False


@dataclass
class HHAction:
    """A single action in a parsed hand."""
    player: str
    action: str      # fold, check, call, bet, raise, all_in, post_blind
    amount: float = 0.0
    total: float = 0.0  # "raises to" total (for raises)
    street: str = "preflop"


@dataclass
class HHHand:
    """A fully parsed hand from hand history."""
    hand_id: str = ""
    timestamp: str = ""
    table_name: str = ""
    max_seats: int = 6
    small_blind: float = 0.0
    big_blind: float = 0.0
    currency: str = "USD"

    # Players
    players: list = field(default_factory=list)       # [HHPlayer, ...]
    dealer_seat: int = 0
    hero_name: str = ""

    # Actions per street
    preflop_actions: list = field(default_factory=list)   # [HHAction, ...]
    flop_actions: list = field(default_factory=list)
    turn_actions: list = field(default_factory=list)
    river_actions: list = field(default_factory=list)

    # Community cards
    flop: list = field(default_factory=list)      # ['Js', 'Ts', '3d']
    turn_card: str = ""                            # '8h'
    river_card: str = ""                           # 'Ac'

    # Results
    total_pot: float = 0.0
    rake: float = 0.0
    winners: list = field(default_factory=list)    # [(name, amount), ...]
    showdown_hands: dict = field(default_factory=dict)  # {name: [card, card]}
    board: list = field(default_factory=list)      # Final board from SUMMARY

    @property
    def all_actions(self) -> list:
        return (self.preflop_actions + self.flop_actions +
                self.turn_actions + self.river_actions)

    @property
    def community_cards(self) -> list:
        cards = list(self.flop)
        if self.turn_card:
            cards.append(self.turn_card)
        if self.river_card:
            cards.append(self.river_card)
        return cards

    def get_player(self, name: str) -> Optional[HHPlayer]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    def player_actions(self, name: str) -> list:
        return [a for a in self.all_actions if a.player == name]

    def went_to_showdown(self) -> bool:
        return len(self.showdown_hands) > 0

    def player_went_to_showdown(self, name: str) -> bool:
        """Check if player was still active at showdown (didn't fold)."""
        if not self.went_to_showdown():
            return False
        # Player didn't fold = still in at showdown
        folded = any(a.player == name and a.action == "fold"
                     for a in self.all_actions)
        return not folded

    def player_won_at_showdown(self, name: str) -> bool:
        return any(w[0] == name for w in self.winners) and self.went_to_showdown()


# --- Card notation conversion ---
# PokerStars uses [As Kh] while our system uses As, Kh (same format actually)
CARD_RE = re.compile(r'[2-9TJQKA][shdc]')


def _normalize_card(card_str: str) -> str:
    """Normalize a card string from HH format to our format.
    PokerStars: 'As', 'Kh', 'Td' — same as our format.
    """
    if len(card_str) == 2 and card_str[0] in '23456789TJQKA' and card_str[1] in 'shdc':
        return card_str
    return card_str


class HandHistoryParser:
    """Parses PokerStars hand history text into structured HHHand objects."""

    # Section markers
    HAND_START = re.compile(
        r"PokerStars (?:Zoom )?Hand #(\d+):\s+"
        r"Hold'em No Limit \("
        r"[$€]?([\d.]+)/[$€]?([\d.]+)\s*(\w*)\)"
        r"\s*-\s*(.+)"
    )
    TABLE_LINE = re.compile(
        r"Table '([^']+)'\s+(\d+)-max\s+Seat #(\d+) is the button"
    )
    SEAT_LINE = re.compile(
        r"Seat (\d+): (.+?) \([$€]?([\d.]+) in chips\)"
    )
    DEALT_TO = re.compile(
        r"Dealt to (.+?) \[(.+?)\]"
    )
    ACTION_LINE = re.compile(
        r"^(.+?): (folds?|checks?|calls?\s*[$€]?[\d.]*|bets?\s*[$€]?[\d.]*|"
        r"raises?\s*[$€]?[\d.]*\s*to\s*[$€]?[\d.]*|posts? (?:small|big) blind\s*[$€]?[\d.]*|"
        r"posts? the ante\s*[$€]?[\d.]*)\s*(?:and is all-in)?$"
    )
    ALL_IN_MARKER = re.compile(r"and is all-in")
    FLOP_LINE = re.compile(r"\*\*\* FLOP \*\*\* \[(.+?)\]")
    TURN_LINE = re.compile(r"\*\*\* TURN \*\*\* \[.+?\] \[(.+?)\]")
    RIVER_LINE = re.compile(r"\*\*\* RIVER \*\*\* \[.+?\] \[(.+?)\]")
    SHOWDOWN_MARKER = re.compile(r"\*\*\* SHOW DOWN \*\*\*")
    SHOWS_LINE = re.compile(r"^(.+?): shows \[(.+?)\]")
    COLLECTED_LINE = re.compile(r"^(.+?) collected [$€]?([\d.]+) from (?:main |side )?pot")
    SUMMARY_POT = re.compile(r"Total pot [$€]?([\d.]+)(?:.*?Rake [$€]?([\d.]+))?")
    SUMMARY_BOARD = re.compile(r"Board \[(.+?)\]")
    POSTS_BLIND = re.compile(
        r"^(.+?): posts (small|big) blind [$€]?([\d.]+)"
    )

    def parse_hand(self, text: str) -> Optional[HHHand]:
        """Parse a single hand history block into an HHHand object."""
        lines = text.strip().split('\n')
        if not lines:
            return None

        hand = HHHand()
        current_street = "preflop"
        in_showdown = False
        player_names = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Hand header
            m = self.HAND_START.match(line)
            if m:
                hand.hand_id = m.group(1)
                hand.small_blind = float(m.group(2))
                hand.big_blind = float(m.group(3))
                hand.currency = m.group(4) or "USD"
                hand.timestamp = m.group(5).strip()
                continue

            # Table info
            m = self.TABLE_LINE.match(line)
            if m:
                hand.table_name = m.group(1)
                hand.max_seats = int(m.group(2))
                hand.dealer_seat = int(m.group(3))
                continue

            # Seat info
            m = self.SEAT_LINE.match(line)
            if m:
                seat = int(m.group(1))
                name = m.group(2).strip()
                stack = float(m.group(3))
                hand.players.append(HHPlayer(name=name, seat=seat, stack=stack))
                player_names.add(name)
                continue

            # Hole cards dealt to hero
            m = self.DEALT_TO.match(line)
            if m:
                hero_name = m.group(1).strip()
                cards_str = m.group(2)
                cards = CARD_RE.findall(cards_str)
                hand.hero_name = hero_name
                hero = hand.get_player(hero_name)
                if hero:
                    hero.hole_cards = [_normalize_card(c) for c in cards]
                    hero.is_hero = True
                continue

            # Street markers
            m = self.FLOP_LINE.match(line)
            if m:
                current_street = "flop"
                cards = CARD_RE.findall(m.group(1))
                hand.flop = [_normalize_card(c) for c in cards]
                continue

            m = self.TURN_LINE.match(line)
            if m:
                current_street = "turn"
                cards = CARD_RE.findall(m.group(1))
                if cards:
                    hand.turn_card = _normalize_card(cards[0])
                continue

            m = self.RIVER_LINE.match(line)
            if m:
                current_street = "river"
                cards = CARD_RE.findall(m.group(1))
                if cards:
                    hand.river_card = _normalize_card(cards[0])
                continue

            # Showdown marker
            if self.SHOWDOWN_MARKER.match(line):
                in_showdown = True
                continue

            # Showdown: player shows cards
            if in_showdown:
                m = self.SHOWS_LINE.match(line)
                if m:
                    name = m.group(1).strip()
                    cards = CARD_RE.findall(m.group(2))
                    hand.showdown_hands[name] = [_normalize_card(c) for c in cards]
                    continue

            # Collected pot
            m = self.COLLECTED_LINE.match(line)
            if m:
                name = m.group(1).strip()
                amount = float(m.group(2))
                hand.winners.append((name, amount))
                continue

            # Summary section
            m = self.SUMMARY_POT.search(line)
            if m:
                hand.total_pot = float(m.group(1))
                if m.group(2):
                    hand.rake = float(m.group(2))
                continue

            m = self.SUMMARY_BOARD.search(line)
            if m:
                cards = CARD_RE.findall(m.group(1))
                hand.board = [_normalize_card(c) for c in cards]
                continue

            # Blind posts
            m = self.POSTS_BLIND.match(line)
            if m:
                name = m.group(1).strip()
                blind_type = m.group(2)
                amount = float(m.group(3))
                action = HHAction(
                    player=name,
                    action="post_blind",
                    amount=amount,
                    street="preflop"
                )
                hand.preflop_actions.append(action)
                continue

            # Regular actions
            action = self._parse_action(line, current_street, player_names)
            if action:
                if current_street == "preflop":
                    hand.preflop_actions.append(action)
                elif current_street == "flop":
                    hand.flop_actions.append(action)
                elif current_street == "turn":
                    hand.turn_actions.append(action)
                elif current_street == "river":
                    hand.river_actions.append(action)

        if not hand.hand_id:
            return None

        return hand

    def _parse_action(self, line: str, street: str,
                      player_names: set) -> Optional[HHAction]:
        """Parse an action line like 'Hero: raises $1.00 to $1.50'."""
        # Quick check: must contain ': ' and start with a known player
        if ': ' not in line:
            return None

        colon_pos = line.index(': ')
        name = line[:colon_pos].strip()

        if name not in player_names:
            return None

        rest = line[colon_pos + 2:].strip().lower()
        is_all_in = 'and is all-in' in rest
        rest = rest.replace('and is all-in', '').strip()

        # Extract amounts
        amounts = re.findall(r'[$€]?([\d.]+)', rest)

        if rest.startswith('fold'):
            action_type = "fold"
            amount = 0.0
        elif rest.startswith('check'):
            action_type = "check"
            amount = 0.0
        elif rest.startswith('call'):
            action_type = "all_in" if is_all_in else "call"
            amount = float(amounts[0]) if amounts else 0.0
        elif rest.startswith('bet'):
            action_type = "all_in" if is_all_in else "bet"
            amount = float(amounts[0]) if amounts else 0.0
        elif rest.startswith('raise'):
            action_type = "all_in" if is_all_in else "raise"
            # "raises $1.00 to $1.50" — amount is the raise-to total
            if len(amounts) >= 2:
                amount = float(amounts[0])
                total = float(amounts[1])
            elif amounts:
                amount = float(amounts[0])
                total = amount
            else:
                amount = 0.0
                total = 0.0
            return HHAction(player=name, action=action_type, amount=amount,
                            total=total, street=street)
        else:
            return None

        return HHAction(player=name, action=action_type, amount=amount,
                        street=street)

    def parse_file(self, filepath: str) -> list:
        """Parse all hands from a hand history file.

        Returns list of HHHand objects, most recent last.
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except (OSError, IOError) as e:
            print(f"HH Parser: Could not read {filepath}: {e}")
            return []

        return self.parse_text(content)

    def parse_text(self, content: str) -> list:
        """Parse all hands from hand history text content."""
        # Split on hand boundaries (blank line followed by PokerStars Hand #)
        blocks = re.split(r'\n\s*\n(?=PokerStars )', content)
        hands = []
        for block in blocks:
            block = block.strip()
            if not block or not block.startswith('PokerStars'):
                continue
            hand = self.parse_hand(block)
            if hand:
                hands.append(hand)
        return hands


class HandHistoryMonitor:
    """Monitors a hand history folder for new/updated files.

    Runs in a background thread, detects new hands, and calls a callback
    for each new hand parsed.
    """

    def __init__(self, hh_dir: str, callback: Callable[[HHHand], None],
                 poll_interval: float = 2.0, hero_name: str = ""):
        """
        Args:
            hh_dir: Path to the hand history directory.
            callback: Called with each new HHHand as it's detected.
            poll_interval: Seconds between directory checks.
            hero_name: If set, only parse files where this name appears.
        """
        self.hh_dir = hh_dir
        self.callback = callback
        self.poll_interval = poll_interval
        self.hero_name = hero_name
        self.parser = HandHistoryParser()
        self._stop_event = threading.Event()
        self._thread = None
        # Track file positions to only parse new content
        self._file_positions: dict[str, int] = {}
        # Track parsed hand IDs to avoid duplicates
        self._seen_hand_ids: set = set()

    def load_existing(self) -> list:
        """Parse all existing HH files and return all hands.

        Also initializes file positions so the monitor only picks up
        new hands going forward.
        """
        all_hands = []
        if not os.path.isdir(self.hh_dir):
            print(f"HH Monitor: Directory not found: {self.hh_dir}")
            return all_hands

        for filename in sorted(os.listdir(self.hh_dir)):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(self.hh_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                self._file_positions[filepath] = len(content.encode('utf-8', errors='replace'))
            except (OSError, IOError):
                continue

            hands = self.parser.parse_text(content)
            for h in hands:
                if h.hand_id not in self._seen_hand_ids:
                    self._seen_hand_ids.add(h.hand_id)
                    all_hands.append(h)

        print(f"HH Monitor: Loaded {len(all_hands)} existing hands "
              f"from {len(self._file_positions)} files")
        return all_hands

    def start(self):
        """Start monitoring in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"HH Monitor: Watching {self.hh_dir}")

    def stop(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor_loop(self):
        """Main monitoring loop — checks for new content in HH files."""
        while not self._stop_event.is_set():
            try:
                self._check_for_updates()
            except Exception as e:
                print(f"HH Monitor error: {e}")
            self._stop_event.wait(self.poll_interval)

    def _check_for_updates(self):
        """Check all HH files for new content appended since last check."""
        if not os.path.isdir(self.hh_dir):
            return

        for filename in os.listdir(self.hh_dir):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(self.hh_dir, filename)

            try:
                file_size = os.path.getsize(filepath)
            except OSError:
                continue

            last_pos = self._file_positions.get(filepath, 0)
            if file_size <= last_pos:
                continue

            # New content available — read from last position
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    f.seek(last_pos)
                    new_content = f.read()
                self._file_positions[filepath] = file_size
            except (OSError, IOError):
                continue

            if not new_content.strip():
                continue

            # Parse new hands from the appended content
            hands = self.parser.parse_text(new_content)
            for hand in hands:
                if hand.hand_id not in self._seen_hand_ids:
                    self._seen_hand_ids.add(hand.hand_id)
                    try:
                        self.callback(hand)
                    except Exception as e:
                        print(f"HH callback error for hand #{hand.hand_id}: {e}")

    @property
    def hands_parsed(self) -> int:
        return len(self._seen_hand_ids)
