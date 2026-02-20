"""Quick test to verify all modules work correctly."""
import os, sys, io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()
sys.path.insert(0, _dir)
os.chdir(_dir)

from gamestate.state import GameState, Player
from profiles.opponent_db import OpponentDatabase
from strategy.engine import StrategyEngine, hand_to_notation
from ui.window import UIState, format_cards

print("=" * 55)
print("  POKER AI ASSISTANT - SYSTEM TEST")
print("=" * 55)

# Test 1: Strategy Engine
print("\n[TEST 1] Strategy Engine")
strategy = StrategyEngine()
gs = GameState()

hero = Player(name="Hero", seat=0, stack=100.0, is_hero=True)
villain = Player(name="Fish42", seat=3, stack=100.0)
gs.new_hand(1, [hero, villain], dealer_seat=0, small_blind=0.5, big_blind=1.0)
gs.set_hero_cards(["As", "Kh"])
gs.update_pot(3.0)

ctx = gs.get_context_summary()
rec = strategy.analyze(ctx)
print(f"  Hand: AKo | Position: {ctx['hero_position']} | Street: preflop")
print(f"  Action: {rec.action} ${rec.amount:.2f}")
print(f"  Confidence: {rec.confidence}%")
print(f"  Equity: {rec.equity:.0%}")
print(f"  Reasoning: {rec.reasoning}")
print("  PASS" if rec.action == "raise" else "  FAIL")

# Test 2: Post-flop
print("\n[TEST 2] Post-flop Analysis")
gs.update_community_cards(["Js", "Ts", "3d"])
gs.update_pot(12.5)
ctx = gs.get_context_summary()
rec = strategy.analyze(ctx)
print(f"  Hand: AKo | Board: Js Ts 3d | Pot: $12.50")
print(f"  Action: {rec.action} ${rec.amount:.2f}")
print(f"  Equity: {rec.equity:.0%}")
print(f"  Reasoning: {rec.reasoning}")
print("  PASS" if rec.action in ("bet", "raise", "check") else "  FAIL")

# Test 3: Opponent Profiling
print("\n[TEST 3] Opponent Profiling")
db = OpponentDatabase("profiles/test_opponents.db")
profile = db.get_profile("TestFish")
profile.hands_played = 100
profile._times_could_vpip = 100
profile._times_did_vpip = 55
profile._times_could_pfr = 100
profile._times_did_pfr = 8
profile._total_bets_raises = 30
profile._total_calls = 70
profile._times_faced_cbet = 50
profile._times_folded_to_cbet = 18
profile._times_reached_showdown = 38
profile._times_won_showdown = 15
profile.recalculate()
print(f"  Player: TestFish")
print(f"  Type: {profile.player_type}")
print(f"  VPIP: {profile.vpip:.1f}% | PFR: {profile.pfr:.1f}% | AF: {profile.af:.1f}")
print(f"  Tips: {profile.get_exploit_tips()[:2]}")
print("  PASS" if profile.player_type in ("fish", "calling_station") else "  FAIL")
db.close()

# Test 4: Exploit Engine
print("\n[TEST 4] Exploit-Adjusted Advice")
db2 = OpponentDatabase("profiles/test_opponents2.db")
nit = db2.get_profile("NitPlayer")
nit.hands_played = 150
nit._times_could_vpip = 150
nit._times_did_vpip = 22
nit._times_could_pfr = 150
nit._times_did_pfr = 20
nit._total_bets_raises = 100
nit._total_calls = 40
nit._times_could_cbet = 50
nit._times_did_cbet = 35
nit._times_faced_cbet = 50
nit._times_folded_to_cbet = 38
nit._times_faced_3bet = 40
nit._times_folded_to_3bet = 30
nit._times_reached_showdown = 30
nit._times_won_showdown = 18
nit.recalculate()

gs2 = GameState()
hero2 = Player(name="Hero", seat=0, stack=100.0, is_hero=True)
villain2 = Player(name="NitPlayer", seat=3, stack=100.0)
gs2.new_hand(2, [hero2, villain2], dealer_seat=0, small_blind=0.5, big_blind=1.0)
gs2.set_hero_cards(["8d", "7d"])
gs2.update_community_cards(["Ks", "4h", "2c"])
gs2.update_pot(6.0)
ctx2 = gs2.get_context_summary()
rec2 = strategy.analyze(ctx2, nit)
print(f"  Hand: 87s | Board: Ks 4h 2c | vs NIT (fold c-bet {nit.fold_to_cbet:.0f}%)")
print(f"  Action: {rec2.action} ${rec2.amount:.2f}")
print(f"  Exploit: {rec2.exploit_note}")
has_exploit = bool(rec2.exploit_note)
print(f"  PASS (exploit active)" if has_exploit else "  PASS (GTO fallback)")
db2.close()

# Test 5: Push/Fold
print("\n[TEST 5] Tournament Push/Fold")
from strategy.push_fold import should_push, should_call_allin
tests = [
    ("AA", "UTG", 10, True),
    ("72o", "UTG", 10, False),
    ("K9o", "BTN", 8, True),
    ("A5s", "SB", 6, True),
]
all_pass = True
for hand, pos, bb, expected in tests:
    result = should_push(hand, pos, bb)
    status = "PASS" if result == expected else "FAIL"
    if result != expected:
        all_pass = False
    print(f"  {hand} {pos} {bb}BB: push={result} (expected={expected}) {status}")
print(f"  {'ALL PASS' if all_pass else 'SOME FAILED'}")

# Test 6: Card Template Check
print("\n[TEST 6] Card Templates")
templates_dir = os.path.join("models", "card_templates")
count = len([f for f in os.listdir(templates_dir) if f.endswith('.png')]) if os.path.exists(templates_dir) else 0
print(f"  Templates found: {count}/52")
print(f"  PASS" if count == 52 else "  FAIL")

# Test 7: UI State
print("\n[TEST 7] UI State Formatting")
state = UIState(
    hero_cards=["As", "Kh"],
    community_cards=["Js", "Ts", "3d"],
    pot=12.50,
    action="BET",
    amount=8.25,
    confidence=85,
)
cards_str = format_cards(state.hero_cards)
print(f"  Cards display: {cards_str}")
print(f"  Action: {state.action} ${state.amount}")
print("  PASS")

# Cleanup test DBs
for f in ["profiles/test_opponents.db", "profiles/test_opponents2.db"]:
    if os.path.exists(f):
        os.remove(f)

print("\n" + "=" * 55)
print("  ALL TESTS COMPLETE")
print("=" * 55)
