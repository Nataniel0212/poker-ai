"""
Poker AI Assistant — Main entry point.

Ties together all modules:
  1. Screen Capture → reads poker client
  2. Vision/OCR → extracts game state from image
  3. Game State Tracker → maintains hand history & context
  4. Opponent Profiler → tracks and classifies opponents
  5. Strategy Engine → calculates optimal play
  6. LLM Advisor → synthesizes advice
  7. UI → displays recommendations

Usage:
    python main.py                    # Run with default config (screen capture)
    python main.py --camera           # Use camera-based capture
    python main.py --calibrate        # Run table calibration
    python main.py --demo             # Run demo mode with simulated hands
"""

import sys
import io
import time
import random
import argparse
import threading

# Fix Windows console encoding for Swedish characters
if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
elif hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import Config
from capture.screen_capture import ScreenCapture, WindowCapture, CameraCapture, CaptureLoop
from vision.table_reader import TableReader, TableCalibrator, TableRegions
from gamestate.state import GameState, Player, ActionType, Hand, Street
from profiles.opponent_db import OpponentDatabase
from strategy.engine import StrategyEngine, Recommendation
from llm.advisor import PokerAdvisor, AdvisorConfig
from ui.window import PokerUI, ConsoleUI, UIState


def _safe_int(value, default=0):
    """Safely convert a value to int, handling non-numeric strings from LLM."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


class PokerAssistant:
    """Main application — orchestrates all modules."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._setup_modules()
        self.running = False

    def _setup_modules(self):
        """Initialize all modules."""
        import os
        self._last_hero_cards = []
        self._last_community_cards = []
        self._last_pot = 0.0
        self._no_cards_frames = 0  # Counter for frames with no hero cards
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()

        # Game state
        self.game_state = GameState()

        # Opponent database
        db_path = os.path.join(base_dir, self.config.opponent_db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.opponent_db = OpponentDatabase(db_path)

        # Strategy engine
        self.strategy = StrategyEngine()

        # LLM advisor
        advisor_config = AdvisorConfig(
            model_name=self.config.llm_model,
            temperature=self.config.llm_temperature,
        )
        self.advisor = PokerAdvisor(advisor_config)

        # Vision (loaded later when regions are configured)
        self.table_reader = None

        # Capture (initialized on start)
        self.capture = None
        self.capture_loop = None

    def setup_capture(self):
        """Initialize the capture module based on config.

        Uses WindowCapture (Win32 PrintWindow) for PokerStars — captures
        window content directly without needing it visible or in foreground.
        Falls back to mss ScreenCapture if hwnd not available.
        """
        if self.config.capture_mode == "camera":
            self.capture = CameraCapture(self.config.camera_index)
            print("Camera capture initialized")
        else:
            try:
                from calibrate_pokerstars import find_pokerstars_window
                ps = find_pokerstars_window()
                if ps and len(ps) >= 6:
                    title, _, _, _, _, hwnd = ps
                    self.capture = WindowCapture(hwnd)
                    self._ps_hwnd = hwnd
                    w, h = self.capture.get_window_size()
                    print(f"WindowCapture initialized: {w}x{h} (PrintWindow API)")
                    print(f"  Window: \"{title}\"")
                else:
                    raise RuntimeError("PokerStars not found")
            except Exception as e:
                print(f"WindowCapture failed ({e}), falling back to mss")
                region = self.config.screen_region
                self.capture = ScreenCapture(region)
                self._ps_hwnd = None

        self.capture_loop = CaptureLoop(self.capture, self.config.capture_fps)

    def setup_vision(self):
        """Initialize the vision module.

        Captures a frame via WindowCapture (or mss fallback), runs
        auto_detect_layout to find card positions, and stores the
        current window size for resize detection.
        """
        import os
        regions = TableRegions()

        try:
            from calibrate_pokerstars import auto_detect_layout, calculate_regions

            # Capture initial frame using our capture module (already set up)
            frame = self.capture.capture() if self.capture else None

            if frame is not None and frame.size > 0:
                fh, fw = frame.shape[:2]
                self._last_window_size = (fw, fh)
                print(f"  Frame: {fw}x{fh}")

                # Try card-based auto-detection
                calc = auto_detect_layout(frame)
                if calc is None:
                    print("  Inga kort synliga — anvander fallback-proportioner")
                    print("  (Regioner kalibreras automatiskt nar kort syns)")
                    calc = calculate_regions(0, 0, fw, fh)
                    self._needs_recalibration = True
                else:
                    print(f"  Auto-detekterade kort! Card size: {calc.get('_card_size')}")
                    self._needs_recalibration = False

                regions = self._calc_to_regions(calc)
                print(f"  Hero card 1: {regions.hero_card_1}")
                print(f"  Community[0]: {regions.community_cards[0]}")
                print(f"  Pot text: {regions.pot_text}")
            else:
                print("Could not capture initial frame — using saved regions")
                self._last_window_size = (0, 0)
                if os.path.exists(self.config.regions_file):
                    calibrator = TableCalibrator()
                    regions = calibrator.load_regions(self.config.regions_file)
        except ImportError:
            self._last_window_size = (0, 0)
            if os.path.exists(self.config.regions_file):
                calibrator = TableCalibrator()
                regions = calibrator.load_regions(self.config.regions_file)
                print(f"Loaded table regions from {self.config.regions_file}")
            else:
                print("No table regions configured. Run with --calibrate first.")

        self.table_reader = TableReader(
            template_dir=self.config.template_dir,
            regions=regions,
            tesseract_cmd=self.config.tesseract_cmd,
        )

    def _calc_to_regions(self, calc: dict) -> TableRegions:
        """Convert auto_detect_layout dict to TableRegions object."""
        regions = TableRegions()
        regions.hero_card_1 = tuple(calc["hero_card_1"])
        regions.hero_card_2 = tuple(calc["hero_card_2"])
        regions.community_cards = [tuple(c) for c in calc["community_cards"]]
        regions.pot_text = tuple(calc["pot_text"])
        regions.player_regions = calc["player_regions"]
        regions.action_buttons_area = tuple(calc.get("action_buttons_area", (0, 0, 0, 0)))
        regions.dealer_button_search_area = tuple(calc.get("dealer_button_search_area", (0, 0, 0, 0)))
        regions.dealer_button_regions = [tuple(r) for r in calc.get("dealer_button_regions", [])]
        return regions

    def recalibrate_if_resized(self, frame: np.ndarray) -> bool:
        """Check if window size changed and re-run layout detection if so.

        Returns True if regions were recalculated.
        """
        if frame is None or frame.size == 0:
            return False
        fh, fw = frame.shape[:2]
        current_size = (fw, fh)

        if current_size == self._last_window_size:
            return False

        print(f"  Window resized: {self._last_window_size} -> {current_size}")
        self._last_window_size = current_size

        try:
            from calibrate_pokerstars import auto_detect_layout, calculate_regions
            calc = auto_detect_layout(frame)
            if calc is None:
                calc = calculate_regions(0, 0, fw, fh)
                print("  Resize: using fallback proportions")
            else:
                print(f"  Resize: auto-detected card layout")
            self.table_reader.regions = self._calc_to_regions(calc)
            return True
        except ImportError:
            return False

    def calibrate(self):
        """Run interactive table calibration."""
        self.setup_capture()
        frame = self.capture.capture()

        calibrator = TableCalibrator()
        regions = calibrator.calibrate_from_frame(frame)

        import os
        os.makedirs(os.path.dirname(self.config.regions_file) or ".", exist_ok=True)
        calibrator.save_regions(self.config.regions_file)

        print("\nCalibration complete! Regions saved.")
        self.capture.release()

    def process_frame(self, frame) -> UIState:
        """Process a single frame through the entire pipeline.

        Returns a UIState ready to be displayed.
        """
        # 1. Read the table
        reading = self.table_reader.read_table(frame)

        # Sticky state: keep last known values when OCR misses
        # Only accept hero cards if we see exactly 2 distinct cards
        if (len(reading.hero_cards) == 2
                and reading.hero_cards[0] != reading.hero_cards[1]):
            hero_cards = reading.hero_cards
        else:
            hero_cards = self._last_hero_cards
        community_cards = reading.community_cards or self._last_community_cards
        pot = reading.pot if reading.pot > 0 else self._last_pot

        # Remove duplicate cards from community (OCR artifact)
        if community_cards:
            seen = set()
            deduped = []
            for c in community_cards:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            community_cards = deduped

        # Remove community cards that duplicate hero cards
        if hero_cards and community_cards:
            hero_set = set(hero_cards)
            community_cards = [c for c in community_cards if c not in hero_set]

        # 2. Detect new hand and ensure a hand exists
        #
        # New hand signals:
        #   a) Hero cards changed (we see 2 new distinct cards)
        #   b) Board disappeared (community went from cards to empty)
        #      + no hero cards visible (between hands)
        #
        new_hand_detected = False

        if not self.game_state.current_hand:
            if hero_cards or reading.players or pot > 0:
                self._create_hand_from_reading(reading)
        else:
            # Signal (a): hero cards changed
            if (len(reading.hero_cards) == 2
                    and reading.hero_cards[0] != reading.hero_cards[1]):
                hero = self.game_state.current_hand.get_hero()
                old_cards = hero.hole_cards if hero else None
                if old_cards and set(old_cards) != set(reading.hero_cards):
                    new_hand_detected = True

            # Signal (b): board disappeared — community was non-empty,
            # now empty, and no hero cards visible (= between hands)
            if (self._last_community_cards
                    and not reading.community_cards
                    and len(reading.hero_cards) != 2):
                self._no_cards_frames += 1
                # Require 3 consecutive empty frames to avoid OCR flicker
                if self._no_cards_frames >= 3:
                    new_hand_detected = True
            else:
                self._no_cards_frames = 0

            if new_hand_detected:
                # Record opponent stats from the completed hand
                self._record_completed_hand()
                # Reset sticky state
                self._last_hero_cards = reading.hero_cards if len(reading.hero_cards) == 2 else []
                self._last_community_cards = []
                self._last_pot = 0.0
                self._no_cards_frames = 0
                hero_cards = self._last_hero_cards
                community_cards = []
                pot = 0.0
                if len(reading.hero_cards) == 2:
                    self._create_hand_from_reading(reading)

        # Update sticky cache AFTER new-hand detection
        # (so detection can compare against previous hand's state)
        if (len(reading.hero_cards) == 2
                and reading.hero_cards[0] != reading.hero_cards[1]):
            self._last_hero_cards = reading.hero_cards
        if reading.community_cards:
            self._last_community_cards = reading.community_cards
        if reading.pot > 0:
            self._last_pot = reading.pot

        if community_cards:
            self.game_state.update_community_cards(community_cards)

        if pot > 0:
            self.game_state.update_pot(pot)

        if hero_cards:
            self.game_state.set_hero_cards(hero_cards)

        # 3. Get game context
        ctx = self.game_state.get_context_summary()

        # 4. Get opponent profile (for heads-up or main villain)
        villain_profile = None
        if self.game_state.current_hand:
            villain = self.game_state.current_hand.get_villain()
            if villain:
                villain_profile = self.opponent_db.get_profile(villain.name)

        # 5. Get strategy recommendation
        rec = self.strategy.analyze(ctx, villain_profile)

        # 6. Optionally enhance with LLM
        rec_dict = {
            "action": rec.action,
            "amount": rec.amount,
            "confidence": rec.confidence,
            "equity": rec.equity,
            "pot_odds": rec.pot_odds,
            "ev": rec.ev,
            "reasoning": rec.reasoning,
            "if_raised": rec.if_raised,
            "if_called": rec.if_called,
            "exploit_note": rec.exploit_note,
        }

        if self.config.llm_enabled and self.advisor.is_available():
            opp_dict = None
            if villain_profile:
                opp_dict = {
                    "player_type": villain_profile.player_type,
                    "hands_played": villain_profile.hands_played,
                    "vpip": villain_profile.vpip,
                    "pfr": villain_profile.pfr,
                    "af": villain_profile.af,
                    "fold_to_cbet": villain_profile.fold_to_cbet,
                    "fold_to_three_bet": villain_profile.fold_to_three_bet,
                    "wtsd": villain_profile.wtsd,
                    "exploit_tips": villain_profile.get_exploit_tips(),
                }
            llm_advice = self.advisor.get_advice(ctx, rec_dict, opp_dict)
            # Merge LLM advice (it may override some fields)
            rec_dict.update(llm_advice)

        # 7. Build UI state (use sticky values for stable display)
        ui_state = UIState(
            hand_number=self.game_state.hands_played,
            hero_cards=hero_cards,
            community_cards=community_cards,
            pot=pot,
            action=rec_dict.get("action", ""),
            amount=rec_dict.get("amount", 0),
            confidence=_safe_int(rec_dict.get("confidence", 0)),
            reasoning=rec_dict.get("reasoning", ""),
            if_raised=rec_dict.get("if_raised", ""),
            if_called=rec_dict.get("if_called", ""),
            exploit_note=rec_dict.get("exploit_note", ""),
            equity=rec_dict.get("equity", 0),
            pot_odds=rec_dict.get("pot_odds", 0),
            ev=rec_dict.get("ev", 0),
            session_profit=self.game_state.session_profit,
            hands_played=self.game_state.hands_played,
        )

        # Add opponent info
        if villain_profile:
            ui_state.opponent_name = villain_profile.name
            ui_state.opponent_type = villain_profile.player_type
            ui_state.opponent_hands = villain_profile.hands_played
            ui_state.opponent_vpip = villain_profile.vpip
            ui_state.opponent_pfr = villain_profile.pfr
            ui_state.opponent_af = villain_profile.af
            ui_state.opponent_fold_cbet = villain_profile.fold_to_cbet
            ui_state.opponent_three_bet = villain_profile.three_bet
            ui_state.opponent_fold_3bet = villain_profile.fold_to_three_bet
            ui_state.opponent_wtsd = villain_profile.wtsd
            ui_state.opponent_tips = villain_profile.get_exploit_tips()

        return ui_state

    def _record_completed_hand(self):
        """Record stats for all villains when a hand completes.

        Called just before creating a new hand, using the current hand's
        action history to update opponent profiles.
        """
        hand = self.game_state.current_hand
        if not hand or not hand.actions:
            return

        hero = hand.get_hero()
        if not hero:
            return

        for player in hand.players:
            if player.is_hero:
                continue
            if not player.name or player.name.startswith("Player_"):
                continue  # Skip unnamed/fallback players

            # Collect this player's actions as dicts (format record_hand expects)
            player_actions = [
                {
                    "action": a.action_type.value,
                    "street": a.street.value,
                    "amount": a.amount,
                }
                for a in hand.actions
                if a.player_name == player.name
            ]

            if not player_actions:
                continue  # Player had no actions, skip

            # Determine if preflop raiser
            was_pfr = any(
                a["action"] in ("raise", "all_in")
                for a in player_actions
                if a["street"] == "preflop"
            )

            # We can't reliably detect showdown from vision alone,
            # so assume went_to_showdown if player was still active at river
            went_to_sd = (player.is_active
                          and hand.street in (Street.RIVER, Street.SHOWDOWN))

            self.opponent_db.record_hand(
                player_name=player.name,
                actions=player_actions,
                went_to_showdown=went_to_sd,
                won_at_showdown=False,  # Can't determine winner from vision
                was_preflop_raiser=was_pfr,
            )

    def _create_hand_from_reading(self, reading):
        """Create a new hand from vision data.

        Identifies hero by matching against the hero card region seat
        (typically seat 0 in calibrated regions, but falls back to
        the player with the lowest seat index if unclear).
        """
        players = []
        hero_seat = self.config.hero_seat  # Default hero seat from config

        for i, p in enumerate(reading.players):
            seat = p.get("seat", i)
            player = Player(
                name=p.get("name", f"Player_{seat}"),
                seat=seat,
                stack=p.get("stack", 0),
                is_hero=(seat == hero_seat),
            )
            players.append(player)

        # If no player was marked as hero, find by seat or create fallback
        if not any(p.is_hero for p in players):
            if players:
                # Mark the player closest to hero_seat as hero
                players.sort(key=lambda p: abs(p.seat - hero_seat))
                players[0].is_hero = True
            else:
                players = [Player(name="Hero", seat=hero_seat, stack=100, is_hero=True)]

        self.game_state.new_hand(
            hand_number=self.game_state.hands_played + 1,
            players=players,
            dealer_seat=reading.dealer_seat if reading.dealer_seat >= 0 else 0,
            small_blind=self.config.default_sb,
            big_blind=self.config.default_bb,
        )

    def run_demo(self):
        """Run a demo with simulated poker hands to test the system."""
        print("\n=== POKER AI ASSISTANT -- DEMO MODE ===\n")

        # Simulate some hands
        demo_hands = [
            {
                "hero_cards": ["As", "Kh"],
                "community": [],
                "pot": 3.0,
                "position": "BTN",
                "bb": 1.0,
                "villain_name": "FishPlayer42",
                "villain_type": "fish",
            },
            {
                "hero_cards": ["Jd", "Td"],
                "community": ["Qs", "9h", "3c"],
                "pot": 12.5,
                "position": "CO",
                "bb": 1.0,
                "villain_name": "TightNit99",
                "villain_type": "nit",
            },
            {
                "hero_cards": ["7h", "7s"],
                "community": ["Ah", "Kd", "7c", "2s"],
                "pot": 24.0,
                "position": "BB",
                "bb": 1.0,
                "villain_name": "AggroManiac",
                "villain_type": "maniac",
            },
        ]

        for i, hand in enumerate(demo_hands):
            print(f"\n{'='*50}")
            print(f"  DEMO HAND #{i+1}")
            print(f"{'='*50}")

            # Set up game state
            hero = Player(name="Hero", seat=0, stack=100.0, is_hero=True)
            villain = Player(name=hand["villain_name"], seat=3, stack=100.0)

            self.game_state.new_hand(
                hand_number=i + 1,
                players=[hero, villain],
                dealer_seat=0 if hand["position"] == "BTN" else 3,
                small_blind=hand["bb"] / 2,
                big_blind=hand["bb"],
            )

            self.game_state.set_hero_cards(hand["hero_cards"])
            if hand["community"]:
                self.game_state.update_community_cards(hand["community"])
            self.game_state.update_pot(hand["pot"])

            # Create a mock opponent profile
            profiles = {
                "fish": {"vpip": 52, "pfr": 8, "af": 0.8, "fold_to_cbet": 35,
                         "fold_to_three_bet": 45, "wtsd": 38, "hands": 87},
                "nit": {"vpip": 14, "pfr": 12, "af": 2.5, "fold_to_cbet": 72,
                        "fold_to_three_bet": 78, "wtsd": 22, "hands": 204},
                "maniac": {"vpip": 45, "pfr": 35, "af": 4.2, "fold_to_cbet": 28,
                           "fold_to_three_bet": 32, "wtsd": 30, "hands": 156},
            }

            profile = self.opponent_db.get_profile(hand["villain_name"])
            p = profiles[hand["villain_type"]]
            profile.hands_played = p["hands"]
            profile._times_could_vpip = 100
            profile._times_did_vpip = int(p["vpip"])
            profile._times_could_pfr = 100
            profile._times_did_pfr = int(p["pfr"])
            profile._total_bets_raises = int(p["af"] * 50)
            profile._total_calls = 50
            profile._times_could_cbet = 50
            profile._times_did_cbet = 25
            profile._times_faced_cbet = 50
            profile._times_folded_to_cbet = int(p["fold_to_cbet"] / 2)
            profile._times_faced_3bet = 50
            profile._times_folded_to_3bet = int(p["fold_to_three_bet"] / 2)
            profile._times_reached_showdown = int(p["wtsd"])
            profile._times_won_showdown = int(p["wtsd"] * 0.5)
            profile.recalculate()

            # Get strategy recommendation
            ctx = self.game_state.get_context_summary()
            rec = self.strategy.analyze(ctx, profile)

            # Build UI state
            ui_state = UIState(
                hand_number=i + 1,
                hero_cards=hand["hero_cards"],
                community_cards=hand["community"],
                pot=hand["pot"],
                action=rec.action,
                amount=rec.amount,
                confidence=rec.confidence,
                reasoning=rec.reasoning,
                if_raised=rec.if_raised,
                if_called=rec.if_called,
                exploit_note=rec.exploit_note,
                equity=rec.equity,
                pot_odds=rec.pot_odds,
                ev=rec.ev,
                opponent_name=hand["villain_name"],
                opponent_type=profile.player_type,
                opponent_hands=profile.hands_played,
                opponent_vpip=profile.vpip,
                opponent_pfr=profile.pfr,
                opponent_af=profile.af,
                opponent_fold_cbet=profile.fold_to_cbet,
                opponent_three_bet=profile.three_bet,
                opponent_fold_3bet=profile.fold_to_three_bet,
                opponent_wtsd=profile.wtsd,
                opponent_tips=profile.get_exploit_tips(),
                session_profit=random.uniform(-5, 15),
                hands_played=i + 1,
            )

            # Print to console (demo always uses console output)
            from ui.window import format_cards
            print(f"  Kort: {format_cards(ui_state.hero_cards)}")
            if ui_state.community_cards:
                print(f"  Board: {format_cards(ui_state.community_cards)}")
            print(f"  Pot: ${ui_state.pot:.2f}")
            print(f"\n  >>> {ui_state.action.upper()} "
                  f"{'$' + f'{ui_state.amount:.2f}' if ui_state.amount else ''} "
                  f"(Confidence: {ui_state.confidence}%)")
            print(f"  {ui_state.reasoning}")
            if ui_state.exploit_note:
                print(f"  EXPLOIT: {ui_state.exploit_note}")
            print(f"  Equity: {ui_state.equity:.0%} | "
                  f"Pot Odds: {ui_state.pot_odds:.0%} | "
                  f"EV: ${ui_state.ev:+.2f}")
            print(f"  Motstandare: {ui_state.opponent_name} "
                  f"({ui_state.opponent_type})")
            time.sleep(1)

        print("\n\n=== DEMO COMPLETE ===")
        print("Run 'python main.py --calibrate' to set up for real play.")

    def shutdown(self):
        """Clean up resources."""
        if self.capture:
            self.capture.release()
        self.opponent_db.close()
        print("Poker AI Assistant shut down.")


class AssistantWorker(threading.Thread):
    """Background worker that captures frames, processes them, and updates UI.

    Runs capture + vision + strategy in a loop. LLM calls are done
    asynchronously to avoid blocking the capture pipeline.
    """

    def __init__(self, assistant: PokerAssistant, ui: PokerUI):
        super().__init__(daemon=True)
        self.assistant = assistant
        self.ui = ui
        self._stop_event = threading.Event()
        self._last_hero_cards = None
        self._last_advice_time = 0
        self._pending_llm = False
        self._llm_lock = threading.Lock()

    def stop(self):
        self._stop_event.set()

    def _save_debug_overlay(self, frame):
        """Save a debug image with all detection regions drawn on the first frame."""
        import os
        import cv2
        import numpy as np
        overlay = frame.copy()
        regions = self.assistant.table_reader.regions
        h, w = frame.shape[:2]
        print(f"  Debug: frame size = {w}x{h}")

        # Draw hero card regions (green)
        for label, reg in [("H1", regions.hero_card_1), ("H2", regions.hero_card_2)]:
            x, y, rw, rh = reg
            cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 255, 0), 2)
            cv2.putText(overlay, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw community card regions (cyan)
        for i, reg in enumerate(regions.community_cards):
            x, y, rw, rh = reg
            cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (255, 255, 0), 2)
            cv2.putText(overlay, f"C{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw pot region (yellow)
        x, y, rw, rh = regions.pot_text
        cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 255, 255), 2)
        cv2.putText(overlay, "POT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Save
        base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
        debug_dir = os.path.join(base, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        path = os.path.join(debug_dir, "region_overlay.png")
        ok, buf = cv2.imencode('.png', overlay)
        if ok:
            with open(path, 'wb') as f:
                f.write(buf)
            print(f"  Debug overlay saved: {path}")

    def run(self):
        assistant = self.assistant
        config = assistant.config
        print("Worker started — capturing frames...")

        if not assistant.capture_loop:
            print("ERROR: capture_loop not initialized!")
            return

        debug_saved = False

        while not self._stop_event.is_set():
            try:
                frame = assistant.capture_loop.update()
                if frame is None:
                    time.sleep(0.05)
                    continue

                # Check if window was resized — recalculate layout if so
                assistant.recalibrate_if_resized(frame)

                # Save debug overlay on first frame
                if not debug_saved:
                    self._save_debug_overlay(frame)
                    debug_saved = True

                # Disable LLM in process_frame — we handle it async below
                saved_llm = config.llm_enabled
                config.llm_enabled = False
                ui_state = assistant.process_frame(frame)
                config.llm_enabled = saved_llm

                # Anti-detection: only show new advice after random delay
                hero_changed = (ui_state.hero_cards != self._last_hero_cards
                                and ui_state.hero_cards)
                now = time.time()

                if hero_changed:
                    self._last_hero_cards = list(ui_state.hero_cards)
                    delay = random.uniform(config.min_response_delay,
                                           config.max_response_delay)
                    self._last_advice_time = now + delay

                # Async LLM enhancement BEFORE masking (so LLM gets real data)
                if (saved_llm and hero_changed
                        and not self._pending_llm):
                    from dataclasses import replace
                    llm_state = replace(ui_state)  # Copy before masking
                    self._run_llm_async(llm_state)

                # Wait for delay before showing advice (anti-detection)
                if now < self._last_advice_time:
                    # Show cards but hide recommendation
                    ui_state.action = ""
                    ui_state.reasoning = "Analyserar..."
                    ui_state.confidence = 0

                self.ui.update_state(ui_state)

            except Exception as e:
                import traceback
                print(f"Worker error: {e}")
                traceback.print_exc()

            time.sleep(0.1)

        print("Worker stopped.")

    def _run_llm_async(self, ui_state: UIState):
        """Run LLM advice in a separate thread to avoid blocking capture."""
        if not self.assistant.advisor.is_available():
            return

        self._pending_llm = True

        def llm_task():
            try:
                ctx = self.assistant.game_state.get_context_summary()
                if not ctx:
                    return

                rec_dict = {
                    "action": ui_state.action,
                    "amount": ui_state.amount,
                    "confidence": ui_state.confidence,
                    "equity": ui_state.equity,
                    "pot_odds": ui_state.pot_odds,
                    "ev": ui_state.ev,
                    "reasoning": ui_state.reasoning,
                }

                villain_profile = None
                if self.assistant.game_state.current_hand:
                    villain = self.assistant.game_state.current_hand.get_villain()
                    if villain:
                        vp = self.assistant.opponent_db.get_profile(villain.name)
                        villain_profile = {
                            "player_type": vp.player_type,
                            "hands_played": vp.hands_played,
                            "vpip": vp.vpip, "pfr": vp.pfr, "af": vp.af,
                            "fold_to_cbet": vp.fold_to_cbet,
                            "fold_to_three_bet": vp.fold_to_three_bet,
                            "wtsd": vp.wtsd,
                            "exploit_tips": vp.get_exploit_tips(),
                        }

                advice = self.assistant.advisor.get_advice(
                    ctx, rec_dict, villain_profile)

                # Update UI with LLM-enhanced advice
                enhanced = UIState(
                    hand_number=ui_state.hand_number,
                    hero_cards=ui_state.hero_cards,
                    community_cards=ui_state.community_cards,
                    pot=ui_state.pot,
                    action=advice.get("action", ui_state.action),
                    amount=advice.get("amount", ui_state.amount),
                    confidence=_safe_int(advice.get("confidence",
                                                    ui_state.confidence)),
                    reasoning=advice.get("reasoning", ui_state.reasoning),
                    if_raised=advice.get("if_raised", ui_state.if_raised),
                    if_called=advice.get("if_called", ui_state.if_called),
                    exploit_note=advice.get("exploit_note",
                                            ui_state.exploit_note),
                    equity=ui_state.equity,
                    pot_odds=ui_state.pot_odds,
                    ev=ui_state.ev,
                    session_profit=ui_state.session_profit,
                    hands_played=ui_state.hands_played,
                )
                self.ui.update_state(enhanced)
                print(f"  LLM: [{advice.get('action', '?')}] "
                      f"{advice.get('reasoning', '')[:60]}")

            except Exception as e:
                print(f"LLM async error: {e}")
            finally:
                self._pending_llm = False

        t = threading.Thread(target=llm_task, daemon=True)
        t.start()


def main():
    parser = argparse.ArgumentParser(description="Poker AI Assistant")
    parser.add_argument("--camera", action="store_true", help="Use camera capture")
    parser.add_argument("--calibrate", action="store_true", help="Run table calibration")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--console", action="store_true", help="Use console UI (no window)")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM advisor")
    args = parser.parse_args()

    config = Config()
    if args.camera:
        config.capture_mode = "camera"
    if args.model:
        config.llm_model = args.model
    if args.no_llm:
        config.llm_enabled = False
    if args.console:
        config.ui_mode = "console"

    assistant = PokerAssistant(config)

    try:
        if args.calibrate:
            assistant.calibrate()
        elif args.demo:
            assistant.run_demo()
        else:
            _run_live(assistant, config)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        assistant.shutdown()


def _run_live(assistant: PokerAssistant, config: Config):
    """Run live mode with UI window and background capture worker."""
    print("=" * 50)
    print("  POKER AI ASSISTANT — LIVE")
    print("=" * 50)

    # Setup capture and vision
    assistant.setup_capture()
    assistant.setup_vision()

    # Check LLM availability
    if config.llm_enabled:
        if assistant.advisor.is_available():
            print(f"  LLM: {config.llm_model} (aktiv)")
        else:
            print(f"  LLM: {config.llm_model} (ej tillganglig — kors utan)")
            config.llm_enabled = False
    else:
        print("  LLM: avstangd (--no-llm)")

    # Create UI (PyQt6 window or console fallback)
    use_pyqt = False
    if config.ui_mode != "console":
        try:
            from PyQt6.QtWidgets import QApplication
            from PyQt6.QtCore import QTimer
            use_pyqt = True
        except ImportError:
            pass

    if use_pyqt:
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        ui = PokerUI()
        ui.show()

        # Start background worker
        assistant.running = True
        worker = AssistantWorker(assistant, ui)
        worker.start()

        print("\nKor med PyQt6-fonster. Stang fonstret for att avsluta.\n")

        # Run Qt event loop on main thread
        def check_worker():
            if not worker.is_alive():
                app.quit()

        timer = QTimer()
        timer.timeout.connect(check_worker)
        timer.start(1000)

        app.exec()

        # Cleanup
        worker.stop()
        worker.join(timeout=3)
    else:
        # Console mode
        ui = ConsoleUI()
        assistant.running = True
        worker = AssistantWorker(assistant, ui)
        worker.start()

        print("\nKor i konsol-lage. Tryck Ctrl+C for att avsluta.\n")

        try:
            while worker.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            worker.stop()
            worker.join(timeout=3)


if __name__ == "__main__":
    main()
