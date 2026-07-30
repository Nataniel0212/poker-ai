"""Invariant-tester for traningsmotorn.

Det viktiga har ar inte att botarna spelar bra, utan att bordet aldrig ljuger:
chips far inte uppsta ur tomma intet, handen maste alltid ta slut, och potten
maste alltid delas ut i sin helhet.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.bots import Bot, chen_score, make_table
from trainer.table import Hand, Player


def play_full_hand(players, button, sb, bb, bots, rng):
    hand = Hand(players, button, sb, bb, rng)
    guard = 0
    while not hand.finished:
        guard += 1
        if guard > 400:
            raise AssertionError("Handen tog aldrig slut")
        player = hand.next_to_act
        opts = hand.options()
        bot = bots.get(player.name) or Bot("tag", rng)
        kind, amount = bot.decide(hand, player, opts)
        hand.act(kind, amount)
    return hand


class TestEngineInvariants(unittest.TestCase):

    def test_chip_conservation_over_many_hands(self):
        rng = random.Random(1234)
        players, bots = make_table("Bot0", 10000.0, 6, rng)
        # Hero styrs ocksa av en bot i det har testet
        bots["Bot0"] = Bot("tag", rng)

        start_total = sum(p.stack for p in players)

        for i in range(400):
            before = sum(p.stack for p in players)
            hand = play_full_hand(players, i % 6, 50.0, 100.0, bots, rng)
            after = sum(p.stack for p in players)

            self.assertAlmostEqual(
                before, after, places=6,
                msg=f"Chips forsvann/skapades i hand {i}: {before} -> {after}",
            )
            for p in players:
                self.assertGreaterEqual(p.stack, -1e-9, f"{p.name} fick negativ stack")

            # Alla som satsat maste ha fatt sin insats representerad i potten
            self.assertAlmostEqual(
                sum(r.amount for r in hand.results),
                sum(p.total_bet for p in players),
                places=6,
                msg=f"Potten delades inte ut korrekt i hand {i}",
            )

            # Ge tillbaka stackar sa ingen slas ut (cash game-stil)
            for p in players:
                p.stack = 10000.0

        self.assertAlmostEqual(start_total, sum(p.stack for p in players), places=6)

    def test_all_in_and_side_pots(self):
        """Tre spelare med olika stackar — sidopotter maste bli rakt fordelade."""
        rng = random.Random(7)
        for trial in range(200):
            players = [
                Player("Kort", 300.0),
                Player("Mellan", 1500.0),
                Player("Djup", 9000.0),
            ]
            bots = {
                "Kort": Bot("maniac", rng),
                "Mellan": Bot("maniac", rng),
                "Djup": Bot("station", rng),
            }
            total_before = sum(p.stack for p in players)
            hand = play_full_hand(players, trial % 3, 50.0, 100.0, bots, rng)
            total_after = sum(p.stack for p in players)

            self.assertAlmostEqual(total_before, total_after, places=6)
            # En kort stack kan aldrig vinna mer an alla matchat mot honom
            kort = players[0]
            self.assertLessEqual(kort.stack, 900.0 + 1e-6)

    def test_heads_up_button_posts_small_blind(self):
        rng = random.Random(3)
        players = [Player("A", 10000.0), Player("B", 10000.0)]
        hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)
        self.assertEqual(players[0].position, "SB")
        self.assertEqual(players[1].position, "BB")
        self.assertAlmostEqual(players[0].street_bet, 50.0)
        self.assertAlmostEqual(players[1].street_bet, 100.0)
        # Preflop heads-up agerar SB/BTN forst
        self.assertEqual(hand.next_to_act.name, "A")

    def test_six_max_positions_and_first_actor(self):
        rng = random.Random(3)
        players = [Player(f"P{i}", 10000.0) for i in range(6)]
        hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)
        self.assertEqual(players[0].position, "BTN")
        self.assertEqual(players[1].position, "SB")
        self.assertEqual(players[2].position, "BB")
        self.assertEqual(players[3].position, "UTG")
        # UTG agerar forst preflop
        self.assertEqual(hand.next_to_act.name, "P3")

    def test_everyone_folds_to_big_blind(self):
        rng = random.Random(11)
        players = [Player(f"P{i}", 10000.0) for i in range(6)]
        hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)
        for _ in range(4):
            hand.act("fold")
        hand.act("fold")  # SB foldar
        self.assertTrue(hand.finished)
        bb = next(p for p in players if p.position == "BB")
        self.assertAlmostEqual(bb.stack, 10000.0 + 50.0)

    def test_check_around_advances_street(self):
        rng = random.Random(5)
        players = [Player(f"P{i}", 10000.0) for i in range(3)]
        hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)
        # Preflop: BTN callar, SB callar, BB checkar
        hand.act("call")
        hand.act("call")
        hand.act("check")
        self.assertEqual(hand.street, "flop")
        self.assertEqual(len(hand.board), 3)

        # Flop: alla checkar -> turn
        hand.act("check")
        hand.act("check")
        hand.act("check")
        self.assertEqual(hand.street, "turn")
        self.assertEqual(len(hand.board), 4)

    def test_min_raise_is_enforced(self):
        rng = random.Random(5)
        players = [Player(f"P{i}", 10000.0) for i in range(3)]
        hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)
        opts = hand.options()
        self.assertAlmostEqual(opts.min_raise_to, 200.0)  # bet 100 + min raise 100
        with self.assertRaises(ValueError):
            hand.act("raise", 150.0)

    def test_reraise_reopens_action(self):
        rng = random.Random(9)
        players = [Player(f"P{i}", 10000.0) for i in range(3)]
        hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)
        hand.act("raise", 300.0)   # BTN oppnar
        hand.act("call")           # SB callar
        hand.act("raise", 900.0)   # BB 3-bettar
        # BTN maste fa agera igen
        self.assertEqual(hand.next_to_act.name, "P0")
        hand.act("call")
        self.assertEqual(hand.next_to_act.name, "P1")  # SB maste ocksa svara

    def test_chen_score_sanity(self):
        self.assertAlmostEqual(chen_score("As", "Ah"), 20.0)
        self.assertAlmostEqual(chen_score("Ks", "Kh"), 16.0)
        self.assertGreater(chen_score("As", "Ks"), chen_score("As", "Kh"))
        # 72o ar sampsta handen och far negativ Chen-poang — det ar korrekt
        self.assertLess(chen_score("7s", "2h"), chen_score("Js", "Ts"))
        self.assertLess(chen_score("7s", "2h"), chen_score("2s", "2h"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
