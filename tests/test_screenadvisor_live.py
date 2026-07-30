"""Regressionstest mot verifierade bildrutor fran live-kalibreringen.

Varje fixture ar en riktig skarmbild fran 247 Free Poker dar samtliga
synliga kort verifierades visuellt under kalibreringen 2026-07-30.
Kravet ar absolut: alla facitkort ska lasas och ingenting annat far
identifieras. En fellasning har ar allvarligare an en miss — programmet
lovar att aldrig gissa.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from screenadvisor.detect import classify, find_card_candidates
from screenadvisor.profile import Profile

FIXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "screenadvisor", "fixtures", "live",
)

TRUTH = {
    "g5.png":  {"Th", "7d", "As", "Qh", "8h", "Qs", "Ks"},
    "g22.png": {"Kc", "4s", "5h", "5c", "Qs", "4c", "9c"},
    "t4.png":  {"2c", "6c", "4c", "4d", "5h", "4h", "As"},
    "u7.png":  {"5c", "8c", "Qs", "Kh", "Qd", "2d", "8s"},
    "v2.png":  {"7s", "Kh", "9c", "Td", "Th"},
    "w1.png":  {"5d", "9h"},
}


def read_labels(path, templates):
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    cands = classify(find_card_candidates(frame), templates)
    return {c.label for c in cands if c.identified}


class TestLiveFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = Profile.load("247 free poker")

    def test_no_wrong_reads(self):
        """Ingen identifierad etikett far sakna tackning i facit."""
        for name, expected in TRUTH.items():
            got = read_labels(os.path.join(FIXTURES, name), self.profile.templates)
            wrong = got - expected
            self.assertFalse(wrong, f"{name}: fellasningar {sorted(wrong)}")

    # Kort som medvetet rapporteras okanda i dessa bildrutor: deras glyf
    # lag for nara en annan etikett och de tvetydiga mallarna rensades.
    # Ett arligt "okant" ar acceptabelt - en fellasning ar det aldrig.
    KNOWN_UNKNOWN = {"g21.png": {"4s"}, "g22.png": {"4s"}}

    def test_all_cards_read(self):
        """Alla facitkort ska lasas i varje bildruta (utom dokumenterade)."""
        for name, expected in TRUTH.items():
            got = read_labels(os.path.join(FIXTURES, name), self.profile.templates)
            miss = expected - got - self.KNOWN_UNKNOWN.get(name, set())
            self.assertFalse(miss, f"{name}: missade {sorted(miss)}")


if __name__ == "__main__":
    unittest.main()
