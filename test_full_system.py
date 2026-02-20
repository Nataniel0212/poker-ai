"""Full system test with LLM integration."""
import os, sys, io, json, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()
sys.path.insert(0, _dir)
os.chdir(_dir)

from gamestate.state import GameState, Player
from profiles.opponent_db import OpponentDatabase
from strategy.engine import StrategyEngine
from llm.advisor import PokerAdvisor, AdvisorConfig
from ui.window import UIState, PokerUI

print("=" * 55)
print("  FULL SYSTEM TEST WITH LLM")
print("=" * 55)

# Setup
strategy = StrategyEngine()
advisor = PokerAdvisor(AdvisorConfig(model_name="poker-ai"))
print(f"\nLLM available: {advisor.is_available()}")

# Test scenario: AKs vs fish on flop
gs = GameState()
hero = Player(name="Hero", seat=0, stack=100.0, is_hero=True)
villain = Player(name="FishyMcFish", seat=3, stack=85.0)
gs.new_hand(1, [hero, villain], dealer_seat=0, small_blind=0.5, big_blind=1.0)
gs.set_hero_cards(["As", "Ks"])
gs.update_community_cards(["Qs", "9h", "3d"])
gs.update_pot(8.0)

# Get game context
ctx = gs.get_context_summary()

# Create opponent profile (fish)
db = OpponentDatabase("profiles/test_full.db")
profile = db.get_profile("FishyMcFish")
profile.hands_played = 92
profile._times_could_vpip = 92
profile._times_did_vpip = 48
profile._times_could_pfr = 92
profile._times_did_pfr = 7
profile._total_bets_raises = 25
profile._total_calls = 65
profile._times_faced_cbet = 40
profile._times_folded_to_cbet = 12
profile._times_faced_3bet = 20
profile._times_folded_to_3bet = 9
profile._times_reached_showdown = 35
profile._times_won_showdown = 12
profile.recalculate()

print(f"\nPlayer profile: {profile.player_type}")
print(f"VPIP: {profile.vpip:.0f}% | PFR: {profile.pfr:.0f}% | AF: {profile.af:.1f}")

# Strategy engine recommendation
rec = strategy.analyze(ctx, profile)
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

print(f"\n--- STRATEGY ENGINE ---")
print(f"Action: {rec.action} ${rec.amount:.2f}")
print(f"Equity: {rec.equity:.0%}")
print(f"Reasoning: {rec.reasoning}")

# LLM advice
if advisor.is_available():
    print(f"\n--- LLM ADVISOR ---")
    print("Querying poker-ai model...")
    t0 = time.time()

    opp_dict = {
        "player_type": profile.player_type,
        "hands_played": profile.hands_played,
        "vpip": profile.vpip,
        "pfr": profile.pfr,
        "af": profile.af,
        "fold_to_cbet": profile.fold_to_cbet,
        "fold_to_three_bet": profile.fold_to_three_bet,
        "wtsd": profile.wtsd,
        "exploit_tips": profile.get_exploit_tips(),
    }

    llm_advice = advisor.get_advice(ctx, rec_dict, opp_dict)
    elapsed = time.time() - t0

    print(f"Response time: {elapsed:.1f}s")
    print(f"LLM Action: {llm_advice.get('action', '?')}")
    print(f"LLM Amount: ${llm_advice.get('amount', 0)}")
    print(f"LLM Confidence: {llm_advice.get('confidence', 0)}%")
    print(f"LLM Reasoning: {llm_advice.get('reasoning', '?')}")
    if llm_advice.get('exploit'):
        print(f"LLM Exploit: {llm_advice['exploit']}")
    if llm_advice.get('if_raised'):
        print(f"LLM If raised: {llm_advice['if_raised']}")
else:
    print("\nLLM not available - skipping LLM test")

# Cleanup
db.close()
if os.path.exists("profiles/test_full.db"):
    os.remove("profiles/test_full.db")

print("\n" + "=" * 55)
print("  TEST COMPLETE")
print("=" * 55)
