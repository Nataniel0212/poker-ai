"""
UI module — separate window displaying poker advice, equity,
opponent stats, and hand history.
"""

import sys
import threading
from dataclasses import dataclass, field

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QFrame, QProgressBar, QScrollArea, QGroupBox,
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import QFont, QColor, QPalette
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False


# Card suit symbols for display
# Use ASCII-safe fallbacks for console; PyQt UI uses Unicode
SUIT_SYMBOLS = {"s": "s", "h": "h", "d": "d", "c": "c"}
SUIT_SYMBOLS_UNICODE = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}
SUIT_COLORS = {"s": "#FFFFFF", "h": "#FF4444", "d": "#4488FF", "c": "#44CC44"}
TYPE_ICONS = {
    "fish": "[FISH]", "shark": "[SHARK]", "maniac": "[MANIAC]", "nit": "[NIT]",
    "calling_station": "[STATION]", "tag": "[TAG]", "lag": "[LAG]",
    "regular": "[REG]", "unknown": "[?]",
}
TYPE_ICONS_UNICODE = {
    "fish": "\U0001f41f", "shark": "\U0001f988", "maniac": "\U0001f412",
    "nit": "\U0001faa8", "calling_station": "\U0001f4de", "tag": "\U0001f3af",
    "lag": "\u26a1", "regular": "\U0001f464", "unknown": "\u2753",
}


def format_card(card: str) -> str:
    """Convert 'As' to 'A♠' with proper symbol."""
    if len(card) < 2:
        return card
    rank = card[0].upper()
    suit = card[1].lower()
    symbol = SUIT_SYMBOLS.get(suit, suit)
    return f"{rank}{symbol}"


def format_cards(cards: list) -> str:
    """Format a list of cards for display."""
    return " ".join(format_card(c) for c in cards)


@dataclass
class UIState:
    """Current state to display in the UI."""
    # Hand info
    hand_number: int = 0
    hero_cards: list = field(default_factory=list)
    community_cards: list = field(default_factory=list)
    pot: float = 0.0

    # Recommendation
    action: str = ""
    amount: float = 0.0
    confidence: int = 0
    reasoning: str = ""
    if_raised: str = ""
    if_called: str = ""
    exploit_note: str = ""

    # Equity
    equity: float = 0.0
    pot_odds: float = 0.0
    ev: float = 0.0

    # Opponent
    opponent_name: str = ""
    opponent_type: str = "unknown"
    opponent_hands: int = 0
    opponent_vpip: float = 0.0
    opponent_pfr: float = 0.0
    opponent_af: float = 0.0
    opponent_fold_cbet: float = 0.0
    opponent_three_bet: float = 0.0
    opponent_fold_3bet: float = 0.0
    opponent_wtsd: float = 0.0
    opponent_tips: list = field(default_factory=list)

    # Session
    session_profit: float = 0.0
    hands_played: int = 0
    hand_history: list = field(default_factory=list)  # List of (result, amount) tuples


if HAS_PYQT:

    class SignalBridge(QObject):
        """Bridge for thread-safe UI updates."""
        update_signal = pyqtSignal(object)

    class PokerUI(QMainWindow):
        """Main poker assistant window."""

        def __init__(self):
            super().__init__()
            self._state_lock = threading.Lock()
            self.state = UIState()
            self.signal_bridge = SignalBridge()
            self.signal_bridge.update_signal.connect(self._apply_state)
            self._setup_ui()

        def _setup_ui(self):
            self.setWindowTitle("Poker AI Assistant")
            self.setMinimumSize(420, 700)
            self.setStyleSheet("""
                QMainWindow { background-color: #1a1a2e; }
                QLabel { color: #e0e0e0; }
                QGroupBox {
                    color: #a0a0c0;
                    border: 1px solid #333366;
                    border-radius: 6px;
                    margin-top: 8px;
                    padding-top: 16px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setSpacing(8)

            # Header
            header = QLabel("POKER AI ASSISTANT")
            header.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
            header.setStyleSheet("color: #7070ff; padding: 5px;")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(header)

            self.hand_label = QLabel("Hand #0")
            self.hand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hand_label.setStyleSheet("color: #666; font-size: 11px;")
            layout.addWidget(self.hand_label)

            # Cards section
            cards_group = QGroupBox("KORT")
            cards_layout = QVBoxLayout(cards_group)

            self.hero_cards_label = QLabel("—")
            self.hero_cards_label.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
            self.hero_cards_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cards_layout.addWidget(self.hero_cards_label)

            self.board_label = QLabel("Board: —")
            self.board_label.setFont(QFont("Consolas", 16))
            self.board_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cards_layout.addWidget(self.board_label)

            self.pot_label = QLabel("Pot: $0.00")
            self.pot_label.setFont(QFont("Consolas", 13))
            self.pot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pot_label.setStyleSheet("color: #ffcc00;")
            cards_layout.addWidget(self.pot_label)

            layout.addWidget(cards_group)

            # Recommendation section
            rec_group = QGroupBox("REKOMMENDATION")
            rec_layout = QVBoxLayout(rec_group)

            self.action_label = QLabel("Väntar...")
            self.action_label.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
            self.action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.action_label.setStyleSheet("""
                background-color: #2a2a4e;
                border-radius: 8px;
                padding: 12px;
                color: #00ff88;
            """)
            rec_layout.addWidget(self.action_label)

            self.confidence_bar = QProgressBar()
            self.confidence_bar.setRange(0, 100)
            self.confidence_bar.setTextVisible(True)
            self.confidence_bar.setFormat("Confidence: %p%")
            self.confidence_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #333;
                    border-radius: 4px;
                    text-align: center;
                    color: white;
                    background-color: #1a1a2e;
                    height: 20px;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff4444, stop:0.5 #ffcc00, stop:1 #00ff88);
                    border-radius: 3px;
                }
            """)
            rec_layout.addWidget(self.confidence_bar)

            self.reasoning_label = QLabel("")
            self.reasoning_label.setWordWrap(True)
            self.reasoning_label.setStyleSheet("color: #b0b0d0; font-size: 11px; padding: 4px;")
            rec_layout.addWidget(self.reasoning_label)

            self.exploit_label = QLabel("")
            self.exploit_label.setWordWrap(True)
            self.exploit_label.setStyleSheet("color: #ff8844; font-size: 11px; padding: 2px;")
            rec_layout.addWidget(self.exploit_label)

            layout.addWidget(rec_group)

            # Equity section
            equity_group = QGroupBox("EQUITY & ODDS")
            equity_layout = QHBoxLayout(equity_group)

            self.equity_label = QLabel("Equity: —")
            self.equity_label.setFont(QFont("Consolas", 11))
            equity_layout.addWidget(self.equity_label)

            self.pot_odds_label = QLabel("Pot Odds: —")
            self.pot_odds_label.setFont(QFont("Consolas", 11))
            equity_layout.addWidget(self.pot_odds_label)

            self.ev_label = QLabel("EV: —")
            self.ev_label.setFont(QFont("Consolas", 11))
            equity_layout.addWidget(self.ev_label)

            layout.addWidget(equity_group)

            # Plan section
            plan_group = QGroupBox("PLAN")
            plan_layout = QVBoxLayout(plan_group)

            self.if_raised_label = QLabel("Om raise: —")
            self.if_raised_label.setWordWrap(True)
            self.if_raised_label.setStyleSheet("color: #cc8888; font-size: 11px;")
            plan_layout.addWidget(self.if_raised_label)

            self.if_called_label = QLabel("Om call: —")
            self.if_called_label.setWordWrap(True)
            self.if_called_label.setStyleSheet("color: #88cc88; font-size: 11px;")
            plan_layout.addWidget(self.if_called_label)

            layout.addWidget(plan_group)

            # Opponent section
            opp_group = QGroupBox("MOTSTÅNDARE")
            opp_layout = QVBoxLayout(opp_group)

            self.opp_name_label = QLabel("—")
            self.opp_name_label.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
            opp_layout.addWidget(self.opp_name_label)

            self.opp_stats_label = QLabel("")
            self.opp_stats_label.setFont(QFont("Consolas", 10))
            self.opp_stats_label.setStyleSheet("color: #a0a0c0;")
            opp_layout.addWidget(self.opp_stats_label)

            self.opp_tips_label = QLabel("")
            self.opp_tips_label.setWordWrap(True)
            self.opp_tips_label.setStyleSheet("color: #ff8844; font-size: 10px;")
            opp_layout.addWidget(self.opp_tips_label)

            layout.addWidget(opp_group)

            # Session section
            session_group = QGroupBox("SESSION")
            session_layout = QVBoxLayout(session_group)

            self.session_label = QLabel("Profit: $0.00 | Hands: 0")
            self.session_label.setFont(QFont("Consolas", 11))
            session_layout.addWidget(self.session_label)

            self.history_label = QLabel("")
            self.history_label.setStyleSheet("color: #888; font-size: 10px;")
            session_layout.addWidget(self.history_label)

            layout.addWidget(session_group)

            layout.addStretch()

        def update_state(self, new_state: UIState):
            """Thread-safe state update. Can be called from any thread."""
            self.signal_bridge.update_signal.emit(new_state)

        def _apply_state(self, state: UIState):
            """Apply state updates to UI elements. Runs on main thread."""
            with self._state_lock:
                self.state = state

            self.hand_label.setText(f"Hand #{state.hand_number}")

            # Cards
            if state.hero_cards:
                self.hero_cards_label.setText(format_cards(state.hero_cards))
            else:
                self.hero_cards_label.setText("—")

            if state.community_cards:
                self.board_label.setText(f"Board: {format_cards(state.community_cards)}")
            else:
                self.board_label.setText("Board: —")

            self.pot_label.setText(f"Pot: ${state.pot:.2f}")

            # Recommendation
            action_text = state.action.upper()
            if state.amount > 0:
                action_text += f" ${state.amount:.2f}"
            self.action_label.setText(action_text or "Väntar...")

            # Color code the action
            colors = {
                "FOLD": "#ff4444", "CHECK": "#aaaaaa", "CALL": "#ffcc00",
                "BET": "#00ff88", "RAISE": "#00ccff",
            }
            color = colors.get(state.action.upper(), "#e0e0e0")
            self.action_label.setStyleSheet(f"""
                background-color: #2a2a4e; border-radius: 8px;
                padding: 12px; color: {color}; font-size: 18px;
            """)

            self.confidence_bar.setValue(state.confidence)
            self.reasoning_label.setText(state.reasoning)
            self.exploit_label.setText(f"⚡ {state.exploit_note}" if state.exploit_note else "")

            # Equity
            self.equity_label.setText(f"Equity: {state.equity:.0%}")
            self.pot_odds_label.setText(f"Pot Odds: {state.pot_odds:.0%}")

            ev_color = "#00ff88" if state.ev >= 0 else "#ff4444"
            self.ev_label.setText(f"EV: ${state.ev:+.2f}")
            self.ev_label.setStyleSheet(f"color: {ev_color};")

            # Plan
            self.if_raised_label.setText(f"Om raise: {state.if_raised or '—'}")
            self.if_called_label.setText(f"Om call: {state.if_called or '—'}")

            # Opponent
            icon = TYPE_ICONS.get(state.opponent_type, "❓")
            self.opp_name_label.setText(
                f"{icon} {state.opponent_name or '—'} ({state.opponent_type.upper()})"
            )
            self.opp_stats_label.setText(
                f"VPIP: {state.opponent_vpip:.0f}% | PFR: {state.opponent_pfr:.0f}% | "
                f"AF: {state.opponent_af:.1f}\n"
                f"Fold C-Bet: {state.opponent_fold_cbet:.0f}% | "
                f"3-Bet: {state.opponent_three_bet:.0f}% | "
                f"WTSD: {state.opponent_wtsd:.0f}%\n"
                f"({state.opponent_hands} händer)"
            )
            if state.opponent_tips:
                self.opp_tips_label.setText("\n".join(f"→ {t}" for t in state.opponent_tips[:3]))
            else:
                self.opp_tips_label.setText("")

            # Session
            profit_color = "#00ff88" if state.session_profit >= 0 else "#ff4444"
            self.session_label.setText(
                f"Profit: ${state.session_profit:+.2f} | Hands: {state.hands_played}"
            )
            self.session_label.setStyleSheet(f"color: {profit_color};")

            # History (last 5)
            if state.hand_history:
                lines = []
                for result, amount in state.hand_history[-5:]:
                    sign = "+" if amount >= 0 else ""
                    lines.append(f"#{result}: {sign}${amount:.2f}")
                self.history_label.setText("\n".join(lines))

class ConsoleUI:
    """Console-based UI — always available, used as fallback or with --console."""

    def __init__(self):
        self.state = UIState()
        self._last_action = None
        print("Konsol-UI aktivt")

    def show(self):
        """No-op for API compatibility with PyQt6 PokerUI."""
        pass

    def update_state(self, new_state: UIState):
        self.state = new_state
        # Only print when action changes (avoid spamming)
        key = (tuple(new_state.hero_cards), new_state.action, new_state.pot)
        if key != self._last_action:
            self._last_action = key
            self._print_state()

    def _print_state(self):
        s = self.state
        print("\n" + "=" * 50)
        print(f"  POKER AI ASSISTANT -- Hand #{s.hand_number}")
        print("=" * 50)
        print(f"  Kort: {format_cards(s.hero_cards) if s.hero_cards else '--'}")
        print(f"  Board: {format_cards(s.community_cards) if s.community_cards else '--'}")
        print(f"  Pot: ${s.pot:.2f}")
        print(f"\n  >>> {s.action.upper()} {'$'+f'{s.amount:.2f}' if s.amount else ''} "
              f"(Confidence: {s.confidence}%)")
        print(f"  {s.reasoning}")
        if s.exploit_note:
            print(f"  EXPLOIT: {s.exploit_note}")
        print(f"\n  Equity: {s.equity:.0%} | Pot Odds: {s.pot_odds:.0%} | EV: ${s.ev:+.2f}")
        if s.if_raised:
            print(f"  Om raise: {s.if_raised}")
        if s.if_called:
            print(f"  Om call: {s.if_called}")
        if s.opponent_name:
            icon = TYPE_ICONS.get(s.opponent_type, "?")
            print(f"\n  Motstandare: {icon} {s.opponent_name} ({s.opponent_type})")
            print(f"  VPIP: {s.opponent_vpip:.0f}% | PFR: {s.opponent_pfr:.0f}% | AF: {s.opponent_af:.1f}")
        print(f"\n  Session: ${s.session_profit:+.2f} ({s.hands_played} hands)")
        print("=" * 50)
        sys.stdout.flush()


if not HAS_PYQT:
    # When PyQt6 is not installed, PokerUI falls back to ConsoleUI
    PokerUI = ConsoleUI
