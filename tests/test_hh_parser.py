"""Test the hand history parser with sample PokerStars hands."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from history.hh_parser import HandHistoryParser

SAMPLE_HH = """
PokerStars Hand #242850173477: Hold'em No Limit ($0.25/$0.50 USD) - 2026/02/20 14:33:05 CET [2026/02/20 8:33:05 ET]
Table 'Lalage IV' 6-max Seat #3 is the button
Seat 1: TightNit99 ($52.35 in chips)
Seat 2: FishPlayer42 ($23.10 in chips)
Seat 3: Nataniel02 ($50.00 in chips)
Seat 4: AggroManiac ($78.50 in chips)
Seat 5: RegularJoe ($45.20 in chips)
Seat 6: SharkMaster ($62.80 in chips)
RegularJoe: posts small blind $0.25
SharkMaster: posts big blind $0.50
*** HOLE CARDS ***
Dealt to Nataniel02 [As Kh]
TightNit99: folds
FishPlayer42: calls $0.50
Nataniel02: raises $1.25 to $1.75
AggroManiac: folds
RegularJoe: folds
SharkMaster: folds
FishPlayer42: calls $1.25
*** FLOP *** [Js Ts 3d]
FishPlayer42: checks
Nataniel02: bets $2.25
FishPlayer42: calls $2.25
*** TURN *** [Js Ts 3d] [Qc]
FishPlayer42: checks
Nataniel02: bets $5.50
FishPlayer42: folds
Uncalled bet ($5.50) returned to Nataniel02
Nataniel02 collected $8.35 from pot
*** SUMMARY ***
Total pot $8.75 | Rake $0.40
Board [Js Ts 3d Qc]
Seat 1: TightNit99 folded before Flop (didn't bet)
Seat 2: FishPlayer42 folded on the Turn
Seat 3: Nataniel02 (button) collected ($8.35)
Seat 4: AggroManiac folded before Flop (didn't bet)
Seat 5: RegularJoe (small blind) folded before Flop
Seat 6: SharkMaster (big blind) folded before Flop


PokerStars Hand #242850173478: Hold'em No Limit ($0.25/$0.50 USD) - 2026/02/20 14:35:12 CET [2026/02/20 8:35:12 ET]
Table 'Lalage IV' 6-max Seat #4 is the button
Seat 1: TightNit99 ($52.35 in chips)
Seat 2: FishPlayer42 ($19.10 in chips)
Seat 3: Nataniel02 ($54.35 in chips)
Seat 4: AggroManiac ($78.50 in chips)
Seat 5: RegularJoe ($44.95 in chips)
Seat 6: SharkMaster ($62.30 in chips)
RegularJoe: posts small blind $0.25
SharkMaster: posts big blind $0.50
*** HOLE CARDS ***
Dealt to Nataniel02 [7h 7s]
TightNit99: folds
FishPlayer42: calls $0.50
Nataniel02: raises $1.25 to $1.75
AggroManiac: raises $4.25 to $6.00
RegularJoe: folds
SharkMaster: folds
FishPlayer42: folds
Nataniel02: calls $4.25
*** FLOP *** [Ah Kd 7c]
Nataniel02: checks
AggroManiac: bets $8.00
Nataniel02: raises $16.00 to $24.00
AggroManiac: calls $16.00
*** TURN *** [Ah Kd 7c] [2s]
Nataniel02: bets $24.35 and is all-in
AggroManiac: calls $24.35
*** RIVER *** [Ah Kd 7c 2s] [9d]
*** SHOW DOWN ***
Nataniel02: shows [7h 7s] (three of a kind, Sevens)
AggroManiac: shows [Ad Ks] (two pair, Aces and Kings)
Nataniel02 collected $107.95 from pot
*** SUMMARY ***
Total pot $109.45 | Rake $1.50
Board [Ah Kd 7c 2s 9d]
Seat 1: TightNit99 folded before Flop (didn't bet)
Seat 2: FishPlayer42 folded before Flop
Seat 3: Nataniel02 showed [7h 7s] and won ($107.95) with three of a kind, Sevens
Seat 4: AggroManiac (button) showed [Ad Ks] and lost with two pair, Aces and Kings
Seat 5: RegularJoe (small blind) folded before Flop
Seat 6: SharkMaster (big blind) folded before Flop
"""


def test_parser():
    parser = HandHistoryParser()
    hands = parser.parse_text(SAMPLE_HH)
    print(f"Parsed {len(hands)} hands")
    assert len(hands) == 2, f"Expected 2 hands, got {len(hands)}"
    print()

    # Hand 1: Simple c-bet pot
    h1 = hands[0]
    print(f"=== Hand 1: #{h1.hand_id} ===")
    assert h1.hand_id == "242850173477"
    assert h1.table_name == "Lalage IV"
    assert h1.max_seats == 6
    assert h1.small_blind == 0.25
    assert h1.big_blind == 0.50
    assert h1.dealer_seat == 3
    assert h1.hero_name == "Nataniel02"
    assert len(h1.players) == 6

    hero = h1.get_player("Nataniel02")
    assert hero is not None
    assert hero.hole_cards == ["As", "Kh"]
    assert hero.is_hero
    assert hero.seat == 3
    assert hero.stack == 50.00

    assert h1.flop == ["Js", "Ts", "3d"]
    assert h1.turn_card == "Qc"
    assert h1.river_card == ""
    assert h1.community_cards == ["Js", "Ts", "3d", "Qc"]
    assert h1.board == ["Js", "Ts", "3d", "Qc"]
    assert h1.total_pot == 8.75
    assert h1.rake == 0.40
    assert ("Nataniel02", 8.35) in h1.winners
    assert not h1.went_to_showdown()

    # Check preflop actions
    pf = h1.preflop_actions
    blind_actions = [a for a in pf if a.action == "post_blind"]
    assert len(blind_actions) == 2
    non_blind = [a for a in pf if a.action != "post_blind"]
    assert non_blind[0].player == "TightNit99" and non_blind[0].action == "fold"
    assert non_blind[1].player == "FishPlayer42" and non_blind[1].action == "call"
    assert non_blind[2].player == "Nataniel02" and non_blind[2].action == "raise"

    # Check flop actions
    assert len(h1.flop_actions) == 3  # check, bet, call
    assert h1.flop_actions[0].action == "check"
    assert h1.flop_actions[1].action == "bet"
    assert h1.flop_actions[1].amount == 2.25

    # Turn: check, bet, fold
    assert len(h1.turn_actions) == 3
    assert h1.turn_actions[2].action == "fold"

    print("  All Hand 1 assertions passed!")
    print()

    # Hand 2: Showdown hand with 3-bet pot
    h2 = hands[1]
    print(f"=== Hand 2: #{h2.hand_id} ===")
    assert h2.hand_id == "242850173478"
    assert h2.dealer_seat == 4

    hero2 = h2.get_player("Nataniel02")
    assert hero2.hole_cards == ["7h", "7s"]

    assert h2.flop == ["Ah", "Kd", "7c"]
    assert h2.turn_card == "2s"
    assert h2.river_card == "9d"
    assert h2.community_cards == ["Ah", "Kd", "7c", "2s", "9d"]
    assert h2.total_pot == 109.45
    assert h2.rake == 1.50

    # Showdown
    assert h2.went_to_showdown()
    assert "Nataniel02" in h2.showdown_hands
    assert h2.showdown_hands["Nataniel02"] == ["7h", "7s"]
    assert h2.showdown_hands["AggroManiac"] == ["Ad", "Ks"]
    assert h2.player_went_to_showdown("Nataniel02")
    assert h2.player_went_to_showdown("AggroManiac")
    assert not h2.player_went_to_showdown("FishPlayer42")
    assert h2.player_won_at_showdown("Nataniel02")
    assert not h2.player_won_at_showdown("AggroManiac")

    # 3-bet: AggroManiac 3-bet over Nataniel02's open
    pf2 = [a for a in h2.preflop_actions if a.action != "post_blind"]
    raises = [a for a in pf2 if a.action == "raise"]
    assert len(raises) == 2  # Nataniel02 open, AggroManiac 3-bet
    assert raises[0].player == "Nataniel02"
    assert raises[1].player == "AggroManiac"

    # All-in on turn
    turn_actions = h2.turn_actions
    all_ins = [a for a in turn_actions if a.action == "all_in"]
    assert len(all_ins) == 1
    assert all_ins[0].player == "Nataniel02"

    print("  All Hand 2 assertions passed!")
    print()

    # Test opponent tracker integration
    from profiles.opponent_db import OpponentDatabase
    from history.opponent_tracker import OpponentTracker
    import tempfile

    db_path = os.path.join(tempfile.gettempdir(), "test_opponents.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    db = OpponentDatabase(db_path)
    tracker = OpponentTracker(db, hero_name="Nataniel02")
    tracker.process_hands(hands)

    # Check FishPlayer42 stats after 2 hands
    fish = db.get_profile("FishPlayer42")
    print(f"FishPlayer42 stats after 2 hands:")
    print(f"  Hands: {fish.hands_played}")
    print(f"  VPIP: {fish.vpip:.1f}%")
    print(f"  PFR: {fish.pfr:.1f}%")
    print(f"  Type: {fish.player_type}")
    assert fish.hands_played == 2
    assert fish.vpip > 0  # Called in both hands
    assert fish.pfr == 0  # Never raised preflop

    # Check AggroManiac stats
    aggro = db.get_profile("AggroManiac")
    print(f"\nAggroManiac stats after 2 hands:")
    print(f"  Hands: {aggro.hands_played}")
    print(f"  VPIP: {aggro.vpip:.1f}%")
    print(f"  PFR: {aggro.pfr:.1f}%")
    print(f"  AF: {aggro.af:.1f}")
    assert aggro.hands_played == 2

    db.close()
    os.remove(db_path)

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_parser()
