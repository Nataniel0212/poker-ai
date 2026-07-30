"""
Live test: Watch for PokerStars hand history files and parse them in real-time.
Run this, then play a few hands in PokerStars. It should detect and parse each hand.

Usage:
    python tests/test_hh_live.py
    python tests/test_hh_live.py --dir "C:/custom/path/to/HandHistory"
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from history.hh_parser import HandHistoryMonitor, HHHand
from history.opponent_tracker import OpponentTracker
from profiles.opponent_db import OpponentDatabase

# Fix console encoding
import io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def on_new_hand(hand: HHHand):
    """Callback for each new hand detected."""
    print(f"\n{'='*60}")
    print(f"  NY HAND: #{hand.hand_id}")
    print(f"{'='*60}")
    print(f"  Bord: {hand.table_name} ({hand.max_seats}-max)")
    print(f"  Blinds: ${hand.small_blind}/${hand.big_blind}")
    print(f"  Hero: {hand.hero_name}")

    hero = hand.get_player(hand.hero_name)
    if hero and hero.hole_cards:
        print(f"  Kort: {' '.join(hero.hole_cards)}")

    print(f"  Spelare:")
    for p in hand.players:
        marker = " <-- HERO" if p.name == hand.hero_name else ""
        print(f"    Seat {p.seat}: {p.name} (${p.stack:.2f}){marker}")

    print(f"  Dealer: seat {hand.dealer_seat}")

    if hand.preflop_actions:
        print(f"\n  Preflop:")
        for a in hand.preflop_actions:
            amt = f" ${a.amount:.2f}" if a.amount else ""
            total = f" -> ${a.total:.2f}" if a.total else ""
            print(f"    {a.player}: {a.action}{amt}{total}")

    if hand.flop:
        print(f"\n  Flop: [{' '.join(hand.flop)}]")
        for a in hand.flop_actions:
            amt = f" ${a.amount:.2f}" if a.amount else ""
            print(f"    {a.player}: {a.action}{amt}")

    if hand.turn_card:
        print(f"\n  Turn: [{hand.turn_card}]")
        for a in hand.turn_actions:
            amt = f" ${a.amount:.2f}" if a.amount else ""
            print(f"    {a.player}: {a.action}{amt}")

    if hand.river_card:
        print(f"\n  River: [{hand.river_card}]")
        for a in hand.river_actions:
            amt = f" ${a.amount:.2f}" if a.amount else ""
            print(f"    {a.player}: {a.action}{amt}")

    if hand.showdown_hands:
        print(f"\n  Showdown:")
        for name, cards in hand.showdown_hands.items():
            print(f"    {name}: [{' '.join(cards)}]")

    if hand.winners:
        print(f"\n  Vinnare:")
        for name, amount in hand.winners:
            print(f"    {name} vann ${amount:.2f}")

    print(f"\n  Pott: ${hand.total_pot:.2f} (rake: ${hand.rake:.2f})")


def find_hh_dir():
    """Auto-detect HandHistory directory."""
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\PokerStars.SE\HandHistory"),
        os.path.expandvars(r"%LOCALAPPDATA%\PokerStars\HandHistory"),
        os.path.join(os.path.expanduser("~"), "Documents", "PokerStars", "HandHistory"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            # Check for user subdirectories
            for sub in os.listdir(candidate):
                subpath = os.path.join(candidate, sub)
                if os.path.isdir(subpath):
                    return subpath
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(description="Live HH parser test")
    parser.add_argument("--dir", type=str, default="",
                        help="Path to HandHistory directory")
    args = parser.parse_args()

    hh_dir = args.dir or find_hh_dir()

    if not hh_dir or not os.path.isdir(hh_dir):
        print("Hand history-mapp hittades inte!")
        print()
        print("Steg:")
        print("1. Oppna PokerStars.SE")
        print("2. Settings > Playing History > Hand History")
        print("3. Bocka i 'Save My Hand History'")
        print("4. Valj mapp och klicka OK")
        print()
        if hh_dir:
            print(f"Sokte i: {hh_dir}")
        else:
            for c in [
                os.path.expandvars(r"%LOCALAPPDATA%\PokerStars.SE\HandHistory"),
                os.path.expandvars(r"%LOCALAPPDATA%\PokerStars\HandHistory"),
            ]:
                print(f"  {c} — {'finns' if os.path.isdir(c) else 'finns inte'}")
        return

    print(f"Overvakar: {hh_dir}")
    print("Spela hander i PokerStars — de dyker upp har!")
    print("Tryck Ctrl+C for att avsluta")
    print()

    # Setup opponent tracking
    import tempfile
    db_path = os.path.join(tempfile.gettempdir(), "hh_test_opponents.db")
    db = OpponentDatabase(db_path)
    tracker = OpponentTracker(db)

    hands_seen = []

    def callback(hand):
        on_new_hand(hand)
        tracker.process_hand(hand)
        hands_seen.append(hand)

        # Show opponent stats after each hand
        hero = hand.hero_name
        print(f"\n  --- Opponent Stats ---")
        for p in hand.players:
            if p.name == hero:
                continue
            profile = db.get_profile(p.name)
            if profile.hands_played > 0:
                print(f"  {p.name}: VPIP={profile.vpip:.0f}% "
                      f"PFR={profile.pfr:.0f}% AF={profile.af:.1f} "
                      f"({profile.hands_played}h, {profile.player_type})")

    monitor = HandHistoryMonitor(
        hh_dir=hh_dir,
        callback=callback,
        poll_interval=1.0,  # Check every second for fast feedback
    )

    # Load existing hands
    existing = monitor.load_existing()
    if existing:
        print(f"Laddat {len(existing)} befintliga hander")
        tracker.process_hands(existing)
        # Show the last hand
        on_new_hand(existing[-1])
    else:
        print("Inga befintliga hander hittades — vantar pa nya...")

    # Start monitoring
    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\nAvslutar. Parsade {len(hands_seen)} nya hander.")
        monitor.stop()
        db.close()


if __name__ == "__main__":
    main()
