"""Glyfnormalisering, mall-arkiv och klassificering.

Hela poangen med den har modulen: **gissa aldrig**. Det gamla OCR-forsoket
svarade "Th" nar det egentligen inte visste, och felen syntes forst som
orimliga rad langt senare. Har returneras `None` sa fort matchningen ar
tveksam, och den som anropar far visa "okant kort" istallet for att ljuga.

En glyf normaliseras till en liten binar ruta. Tva glyfer jamfors med
Jaccard-index (overlapp / union), vilket ar okansligt for storleksskillnader
och ger ett tolkbart matt mellan 0 och 1.
"""

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GLYPH_SIZE = 28          # normaliserad glyfstorlek i pixlar
MIN_SCORE = 0.80         # under detta ar matchningen inte trovardig.
                         # Korrekta traffar mot mallar fran samma skala ligger
                         # pa 0.94-1.00; observerade fellasningar lag 0.64-0.84.
MIN_MARGIN = 0.06        # tvaan maste vara sa mycket samre an ettan

RANKS = "23456789TJQKA"
SUITS = "shdc"
RED_SUITS = ("h", "d")
BLACK_SUITS = ("s", "c")


def normalize(patch: np.ndarray) -> Optional[np.ndarray]:
    """Gor en glyfbild till en binar GLYPH_SIZE-ruta, beskuren till blackets kant.

    Bevarar proportionerna och centrerar — annars skulle ett smalt 'J' och ett
    brett 'Q' bli omojliga att skilja pa efter uppskalning.
    """
    if patch is None or patch.size == 0:
        return None

    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Otsu klarar bade morka och roda glyfer efter graskonvertering
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 2 or h < 2:
        return None
    binary = binary[y:y + h, x:x + w]

    # Skala till att fylla rutan med bevarade proportioner
    scale = (GLYPH_SIZE - 4) / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((GLYPH_SIZE, GLYPH_SIZE), dtype=np.uint8)
    off_x = (GLYPH_SIZE - new_w) // 2
    off_y = (GLYPH_SIZE - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return (canvas > 127).astype(np.uint8)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard-index mellan tva binara glyfer. 1.0 = identiska."""
    if a is None or b is None:
        return 0.0
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


@dataclass
class Match:
    label: str
    score: float
    runner_up: Optional[str] = None
    runner_up_score: float = 0.0

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score

    @property
    def trusted(self) -> bool:
        return self.score >= MIN_SCORE and self.margin >= MIN_MARGIN


class TemplateStore:
    """Inlarda glyfer per sajtprofil.

    Flera varianter per etikett tillats med flit: samma rank kan se olika ut
    beroende pa om kortet ar nagot roterat eller skalat, och da ar det battre
    att spara bada an att snitta ihop dem till en suddig mall.
    """

    MAX_VARIANTS = 12

    def __init__(self):
        self.ranks: Dict[str, List[np.ndarray]] = {}
        self.suits: Dict[str, List[np.ndarray]] = {}

    # ---------- innehall ----------

    def add_rank(self, label: str, glyph: np.ndarray) -> None:
        self._add(self.ranks, label, glyph)

    def add_suit(self, label: str, glyph: np.ndarray) -> None:
        self._add(self.suits, label, glyph)

    def _add(self, bucket: Dict[str, List[np.ndarray]], label: str,
             glyph: np.ndarray) -> None:
        if glyph is None:
            return
        variants = bucket.setdefault(label, [])
        # Spara inte en variant vi i praktiken redan har
        for existing in variants:
            if similarity(existing, glyph) > 0.93:
                return
        variants.append(glyph)
        if len(variants) > self.MAX_VARIANTS:
            variants.pop(0)

    @property
    def known_ranks(self) -> List[str]:
        return sorted(self.ranks, key=lambda r: RANKS.index(r) if r in RANKS else 99)

    @property
    def known_suits(self) -> List[str]:
        return sorted(self.suits)

    def is_complete(self) -> bool:
        return len(self.ranks) == 13 and len(self.suits) == 4

    def missing(self) -> Tuple[List[str], List[str]]:
        return (
            [r for r in RANKS if r not in self.ranks],
            [s for s in SUITS if s not in self.suits],
        )

    # ---------- klassificering ----------

    def match_rank(self, glyph: np.ndarray) -> Optional[Match]:
        return self._match(self.ranks, glyph, None)

    def match_suit(self, glyph: np.ndarray, is_red: Optional[bool] = None) -> Optional[Match]:
        """Farg smalnar av till tva kandidater, vilket gor formvalet mycket sakrare."""
        allowed = None
        if is_red is True:
            allowed = set(RED_SUITS)
        elif is_red is False:
            allowed = set(BLACK_SUITS)
        return self._match(self.suits, glyph, allowed)

    def _match(self, bucket, glyph, allowed) -> Optional[Match]:
        if glyph is None or not bucket:
            return None
        scores: List[Tuple[str, float]] = []
        for label, variants in bucket.items():
            if allowed is not None and label not in allowed:
                continue
            best = max(similarity(v, glyph) for v in variants)
            scores.append((label, best))
        if not scores:
            return None
        scores.sort(key=lambda t: -t[1])
        top_label, top_score = scores[0]
        if len(scores) > 1:
            return Match(top_label, top_score, scores[1][0], scores[1][1])
        return Match(top_label, top_score)

    # ---------- lagring ----------

    def to_dict(self) -> dict:
        def pack(bucket):
            return {
                label: [
                    base64.b64encode(np.packbits(v).tobytes()).decode("ascii")
                    for v in variants
                ]
                for label, variants in bucket.items()
            }
        return {
            "glyph_size": GLYPH_SIZE,
            "ranks": pack(self.ranks),
            "suits": pack(self.suits),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateStore":
        store = cls()
        size = data.get("glyph_size", GLYPH_SIZE)
        total = size * size

        def unpack(packed: str) -> np.ndarray:
            raw = np.frombuffer(base64.b64decode(packed), dtype=np.uint8)
            bits = np.unpackbits(raw)[:total]
            return bits.reshape((size, size)).astype(np.uint8)

        for label, variants in (data.get("ranks") or {}).items():
            store.ranks[label] = [unpack(v) for v in variants]
        for label, variants in (data.get("suits") or {}).items():
            store.suits[label] = [unpack(v) for v in variants]
        return store

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "TemplateStore":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except (ValueError, OSError, KeyError):
            return cls()
