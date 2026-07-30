"""Tester for skarmlasningen, mot riktiga skarmdumpar.

Det viktigaste testet i filen ar `test_unlearned_card_is_never_guessed`. Det
gamla OCR-forsoket svarade alltid nagot — och nar det svarade fel gick felet
rakt in i radgivningen utan att nagon markte det. Har ar kravet det motsatta:
det programmet inte kan lasa sakert ska det saga att det inte kan lasa.
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screenadvisor.advice import build_advice, find_draws, preflop_positions
from screenadvisor.detect import find_card_candidates
from screenadvisor.glyphs import TemplateStore, normalize, similarity
from screenadvisor.reader import read_table

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "screenadvisor", "fixtures")

# Kanda kortpositioner i skarmdumparna (rankglyfens x-koordinat)
KNOWN = {
    "247_kk_755": {240: "Kc", 307: "Kh"},
    "247_kk_716": {245: "Kc", 312: "Kh"},
    "247_72_709": {240: "7h", 307: "2c"},
}


def load(name: str) -> np.ndarray:
    path = os.path.join(FIXTURES, name + ".png")
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def teach(store: TemplateStore, fixture: str, cards=None) -> TemplateStore:
    """Lar in glyferna fran en skarmdump med kant facit."""
    img = load(fixture)
    by_x = {c.rank_mark.x: c for c in find_card_candidates(img)}
    for x, label in KNOWN[fixture].items():
        if cards is not None and label not in cards:
            continue
        cand = by_x[x]
        store.add_rank(label[0], cand.rank_glyph)
        store.add_suit(label[1], cand.suit_glyph)
    return store


def full_store() -> TemplateStore:
    store = TemplateStore()
    teach(store, "247_kk_755")
    teach(store, "247_72_709")
    return store


class TestGlyphs(unittest.TestCase):

    def test_identical_glyphs_match_perfectly(self):
        patch = np.full((20, 16, 3), 240, np.uint8)
        cv2.putText(patch, "K", (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        glyph = normalize(patch)
        self.assertIsNotNone(glyph)
        self.assertAlmostEqual(similarity(glyph, glyph), 1.0)

    def test_different_glyphs_score_low(self):
        store = full_store()
        k_glyph = store.ranks["K"][0]
        seven_glyph = store.ranks["7"][0]
        self.assertLess(similarity(k_glyph, seven_glyph), 0.6)

    def test_blank_patch_returns_none(self):
        self.assertIsNone(normalize(np.full((10, 10, 3), 255, np.uint8)))

    def test_store_survives_save_and_load(self):
        store = full_store()
        path = os.path.join(FIXTURES, "_roundtrip.json")
        try:
            store.save(path)
            loaded = TemplateStore.load(path)
            self.assertEqual(set(loaded.ranks), set(store.ranks))
            self.assertEqual(set(loaded.suits), set(store.suits))
            for label, variants in store.ranks.items():
                for original, restored in zip(variants, loaded.ranks[label]):
                    self.assertAlmostEqual(similarity(original, restored), 1.0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_missing_reports_what_is_not_learned(self):
        store = full_store()
        ranks, suits = store.missing()
        self.assertNotIn("K", ranks)
        self.assertIn("Q", ranks)
        self.assertIn("s", suits)
        self.assertFalse(store.is_complete())


class TestReading(unittest.TestCase):

    def setUp(self):
        self.store = full_store()

    def test_reads_all_fixtures_correctly(self):
        expected = {
            "247_kk_755": ["Kc", "Kh"],
            "247_kk_716": ["Kc", "Kh"],
            "247_72_709": ["7h", "2c"],
        }
        for fixture, hero in expected.items():
            read = read_table(load(fixture), self.store, opponents_override=4)
            self.assertEqual(read.hero, hero, f"fel lasning i {fixture}")
            self.assertEqual(read.board, [], f"falska bordskort i {fixture}")
            self.assertTrue(read.usable, f"{fixture} bedomdes oanvandbar")
            self.assertEqual(read.unknown_cards, 0)

    def test_templates_generalise_across_scale(self):
        """Glyfer inlarda i en storlek maste funka i en annan.

        Spelfonstret kan andra storlek — da far inte lasningen falla ihop.
        """
        store = TemplateStore()
        teach(store, "247_kk_755")          # lar in i 755 px bredd
        read = read_table(load("247_kk_716"), store, opponents_override=4)
        self.assertEqual(read.hero, ["Kc", "Kh"])

    def test_unlearned_card_is_never_guessed(self):
        """En rank som inte lards in far aldrig bli en annan rank."""
        store = TemplateStore()
        teach(store, "247_72_709")          # kan 7 och 2, men inte K
        read = read_table(load("247_kk_755"), store, opponents_override=4)

        self.assertNotIn("Kc", read.hero)   # K ar inte inlard
        for card in read.hero + read.board:
            self.assertNotIn(card[0], ("7", "2"),
                             f"kungen lastes felaktigt som {card}")
        self.assertGreater(read.unknown_cards, 0,
                           "olasta kort rapporterades inte som okanda")
        self.assertFalse(read.usable,
                         "gav rad trots att korten inte kunde lasas")

    def test_empty_store_reads_nothing_and_says_so(self):
        read = read_table(load("247_kk_755"), TemplateStore())
        self.assertEqual(read.hero, [])
        self.assertFalse(read.usable)

    def test_noise_is_not_reported_as_unknown_cards(self):
        """Chipsstaplar och pott-text far inte rakna som olasta kort.

        Annars skulle verktyget staendigt havda att det inte kan lasa bordet,
        trots att alla riktiga kort ar lasta.
        """
        for fixture in KNOWN:
            read = read_table(load(fixture), self.store, opponents_override=4)
            self.assertEqual(read.unknown_cards, 0, f"falskt larm i {fixture}")

    def test_blank_felt_yields_no_cards(self):
        green = np.zeros((300, 400, 3), np.uint8)
        green[:, :] = (40, 110, 30)
        read = read_table(green, self.store)
        self.assertEqual(read.hero, [])
        self.assertEqual(read.unknown_cards, 0)
        self.assertFalse(read.usable)


class TestAdvice(unittest.TestCase):

    def test_premium_hand_playable_everywhere(self):
        self.assertEqual(len(preflop_positions("KK")), 6)

    def test_trash_hand_playable_nowhere(self):
        self.assertEqual(preflop_positions("72o"), [])

    def test_suited_connector_only_from_late_positions(self):
        positions = preflop_positions("54s")
        self.assertNotIn("UTG", positions)
        self.assertIn("BTN", positions)

    def test_flush_draw_is_detected_with_nine_outs(self):
        draws, outs = find_draws(["9h", "8h"], ["Ah", "5h", "2c"])
        self.assertTrue(any("Flushdrag" in d for d in draws))
        self.assertEqual(outs, 9)

    def test_open_ended_straight_draw_has_eight_outs(self):
        draws, outs = find_draws(["9c", "8d"], ["7h", "6s", "2c"])
        self.assertTrue(any("Oppet" in d for d in draws))
        self.assertEqual(outs, 8)

    def test_gutshot_has_four_outs(self):
        draws, outs = find_draws(["9c", "8d"], ["6h", "5s", "2c"])
        self.assertTrue(any("Gutshot" in d for d in draws))
        self.assertEqual(outs, 4)

    def test_no_draws_on_the_river(self):
        draws, outs = find_draws(["9h", "8h"], ["Ah", "5h", "2c", "3d", "Kd"])
        self.assertEqual(draws, [])
        self.assertEqual(outs, 0)

    def test_made_flush_is_not_called_a_draw(self):
        draws, _ = find_draws(["9h", "8h"], ["Ah", "5h", "2h"])
        self.assertFalse(any("Flushdrag" in d for d in draws))

    def test_strong_hand_advises_betting(self):
        advice = build_advice(["As", "Ah"], ["Ad", "7c", "2d"], opponents=1, sims=3000)
        self.assertGreater(advice.equity, 0.85)
        self.assertIn("satsa", advice.headline.lower())

    def test_weak_hand_advises_caution(self):
        advice = build_advice(["7c", "2d"], ["As", "Kh", "Qd"], opponents=3, sims=3000)
        self.assertLess(advice.equity, 0.2)
        self.assertIn("svag", advice.headline.lower())

    def test_required_equity_thresholds_are_correct(self):
        """Att syna en pottstor insats kraver 33 % — grundlaggande pot odds."""
        advice = build_advice(["As", "Ks"], ["Qs", "7c", "2d"], opponents=1, sims=2000)
        needed = [l for l in advice.lines if "pott " in l or l.strip().startswith("pott")]
        joined = " ".join(advice.lines)
        self.assertIn("33%", joined)     # pottstor insats
        self.assertIn("25%", joined)     # halv pott

    def test_no_hero_cards_gives_no_advice(self):
        advice = build_advice([], [], opponents=2)
        self.assertEqual(advice.equity, 0.0)
        self.assertIn("Vantar", advice.headline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
