"""Tester for coachen — ger den ratt rad i spottar dar svaret ar uppenbart?

Om coachen sager fel sak i en solklar situation ar hela traningsverktyget
kontraproduktivt. Darfor testas den mot spottar med entydigt facit.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.cards import FULL_DECK
from trainer.coach import Coach
from trainer.table import Hand, Player


def make_spot(hero_hole, board, street="flop", pot=1000.0, facing_bet=0.0,
              hero_stack=10000.0, villain_stack=10000.0):
    """Bygg en exakt spott att fraga coachen om."""
    rng = random.Random(0)
    players = [
        Player("Du", hero_stack, is_hero=True),
        Player("Bot", villain_stack),
    ]
    hand = Hand(players, button=0, small_blind=50.0, big_blind=100.0, rng=rng)

    hero, villain = players
    hero.hole = list(hero_hole)
    used = set(board) | set(hero_hole)
    villain.hole = [c for c in FULL_DECK if c not in used][:2]

    hand.board = list(board)
    hand.street = street
    hand.actions = []

    # Bygg upp potten jamnt fordelad fran tidigare gator
    each = pot / 2.0
    for p in players:
        p.street_bet = 0.0
        p.total_bet = each

    if facing_bet > 0:
        villain.street_bet = facing_bet
        villain.total_bet += facing_bet
        villain.stack -= facing_bet
        hand.current_bet = facing_bet
        hand.last_raise_size = facing_bet
    else:
        hand.current_bet = 0.0
        hand.last_raise_size = hand.bb

    hand._turn = 0
    hand._acted = set()
    return hand, hero


class TestCoachPreflop(unittest.TestCase):

    def setUp(self):
        self.coach = Coach(sims=3000)

    def _preflop_spot(self, hole, position_button=0, n=6):
        rng = random.Random(42)
        players = [Player("Du", 10000.0, is_hero=True)]
        players += [Player(f"B{i}", 10000.0) for i in range(n - 1)]
        hand = Hand(players, button=position_button, small_blind=50.0,
                    big_blind=100.0, rng=rng)
        hero = players[0]
        used = set(hole)
        pool = [c for c in FULL_DECK if c not in used]
        for i, p in enumerate(players[1:]):
            p.hole = pool[i * 2:i * 2 + 2]
        hero.hole = list(hole)
        # Se till att hjalten ar i tur
        hand._turn = 0
        hand._acted = set()
        return hand, hero

    def test_folding_aces_is_a_mistake(self):
        hand, hero = self._preflop_spot(["As", "Ah"], position_button=0)
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "fold", 0.0)
        self.assertEqual(fb.verdict, "misstag")
        self.assertGreater(fb.equity, 0.4)

    def test_raising_aces_is_correct(self):
        hand, hero = self._preflop_spot(["As", "Ah"], position_button=0)
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "raise", opts.min_raise_to)
        self.assertEqual(fb.verdict, "ratt")

    def test_playing_seven_deuce_from_early_is_a_mistake(self):
        # Hjalten pa BTN-position men handen ar oppelbart ospelbar
        hand, hero = self._preflop_spot(["7c", "2d"], position_button=3)
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "raise", opts.min_raise_to)
        self.assertEqual(fb.verdict, "misstag")
        self.assertIn("range", fb.concept.lower())

    def test_folding_trash_is_correct(self):
        hand, hero = self._preflop_spot(["7c", "2d"], position_button=3)
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "fold", 0.0)
        self.assertEqual(fb.verdict, "ratt")


class TestCoachPostflop(unittest.TestCase):

    def setUp(self):
        self.coach = Coach(sims=4000)

    def test_folding_the_nuts_is_a_big_mistake(self):
        # Stege pa floppen, motstandaren betar litet
        hand, hero = make_spot(
            ["Qs", "Js"], ["As", "Ks", "Ts"], street="flop",
            pot=1000.0, facing_bet=200.0,
        )
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "fold", 0.0)
        self.assertEqual(fb.verdict, "misstag")
        self.assertGreater(fb.equity, 0.9)
        self.assertGreater(fb.ev_loss_bb, 5.0)

    def test_calling_with_odds_beats_folding(self):
        # Flushdrag pa floppen mot liten bet — ska aldrig foldas
        hand, hero = make_spot(
            ["9h", "8h"], ["Ah", "5h", "2c"], street="flop",
            pot=1000.0, facing_bet=200.0,
        )
        opts = hand.options()
        eq = self.coach  # bara for lasbarhet
        fold_fb = self.coach.review(hand, hero, opts, "fold", 0.0)
        self.assertNotEqual(fold_fb.verdict, "ratt")

    def test_calling_huge_bet_with_air_is_a_mistake(self):
        # Fullstandig luft mot en pottstor satsning
        hand, hero = make_spot(
            ["7c", "2d"], ["As", "Kh", "Qd", "Jc", "9s"], street="river",
            pot=1000.0, facing_bet=1000.0,
        )
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "call", 0.0)
        self.assertEqual(fb.verdict, "misstag")
        self.assertEqual(fb.best_label, "fold")

    def test_folding_air_to_big_bet_is_correct(self):
        hand, hero = make_spot(
            ["7c", "2d"], ["As", "Kh", "Qd", "Jc", "9s"], street="river",
            pot=1000.0, facing_bet=1000.0,
        )
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "fold", 0.0)
        self.assertEqual(fb.verdict, "ratt")

    def test_checking_the_nuts_loses_value(self):
        hand, hero = make_spot(
            ["Qs", "Js"], ["As", "Ks", "Ts"], street="flop", pot=1000.0,
        )
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "check", 0.0)
        self.assertEqual(fb.verdict, "misstag")
        self.assertIn("värde", fb.concept.lower())

    def test_betting_the_nuts_is_correct(self):
        hand, hero = make_spot(
            ["Qs", "Js"], ["As", "Ks", "Ts"], street="flop", pot=1000.0,
        )
        opts = hand.options()
        fb = self.coach.review(hand, hero, opts, "raise", 750.0)
        self.assertIn(fb.verdict, ("ratt", "narapa"))

    def test_bet_sizing_is_not_always_maximal(self):
        """Modellen far inte alltid saga 'satsa storsta mojliga'."""
        hand, hero = make_spot(
            ["9c", "9d"], ["9s", "4h", "2c"], street="flop", pot=1000.0,
        )
        opts = hand.options()
        eq = 0.85
        evs = self.coach.ev_table(hand, hero, opts, eq)
        # Med maratt-modellen ska skillnaden mellan halv och hel pott vara liten,
        # inte en jordskredsseger for storsta storleken
        half = evs["bet 1/2 pott"]
        full = evs["bet pott"]
        self.assertLess(abs(full - half) / hand.bb, 6.0)

    def test_verdicts_are_stable_across_runs(self):
        """Samma spott maste fa samma betyg varje gang.

        Equity ar ett Monte Carlo-stickprov. Utan felmarginalen i betygsattningen
        kunde ett och samma beslut bli "ratt" ena gangen och "misstag" nasta —
        det gor coachen omojlig att lita pa.
        """
        spots = [
            (["7c", "2d"], ["As", "Kh", "Qd", "Jc", "9s"], "river", 1000.0, "fold"),
            (["7c", "2d"], ["As", "Kh", "Qd", "Jc", "9s"], "river", 1000.0, "call"),
            (["Qs", "Js"], ["As", "Ks", "Ts"], "flop", 200.0, "fold"),
            (["Qs", "Js"], ["As", "Ks", "Ts"], "flop", 0.0, "check"),
        ]
        for hole, board, street, bet, action in spots:
            verdicts = set()
            for _ in range(8):
                hand, hero = make_spot(hole, board, street=street,
                                       pot=1000.0, facing_bet=bet)
                fb = self.coach.review(hand, hero, hand.options(), action, 0.0)
                verdicts.add(fb.verdict)
            self.assertEqual(
                len(verdicts), 1,
                f"Betyget vippade for {hole} pa {board}: {verdicts}",
            )

    def test_ev_table_marks_fold_as_zero_baseline(self):
        hand, hero = make_spot(
            ["7c", "2d"], ["As", "Kh", "Qd"], street="flop",
            pot=1000.0, facing_bet=500.0,
        )
        opts = hand.options()
        evs = self.coach.ev_table(hand, hero, opts, 0.10)
        self.assertEqual(evs["fold"], 0.0)
        self.assertLess(evs["call"], 0.0)  # 10% equity mot halvpott = forlust


if __name__ == "__main__":
    unittest.main(verbosity=2)
