"""Hittar kortens hornindex i en bild.

Metoden bygger pa en observation om hur de har spelen ritar kort: aven nar
korten overlappar varandra ar hornindexet — rankbokstaven med fargsymbolen
strax under — alltid synligt. Det ar hela poangen med att fjadra korten.

Darfor letar vi inte efter kortens konturer (tva overlappande kort smalter ihop
till en enda vit klump och gar inte att dela palitligt), utan efter just de
glyfparen. Ett par = ett kort.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from screenadvisor.glyphs import Match, TemplateStore, normalize


@dataclass
class Mark:
    """En sammanhangande blackfleck pa en ljus kortyta."""
    x: int
    y: int
    w: int
    h: int
    area: int
    is_red: bool
    face_area: int = 0   # storleken pa den ljusa ytan glyfen ligger pa

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def aspect(self) -> float:
        return self.w / self.h if self.h else 0.0


@dataclass
class CardCandidate:
    """Ett hornindex: rankglyf plus fargglyf under."""
    rank_mark: Mark
    suit_mark: Mark
    rank_glyph: np.ndarray
    suit_glyph: np.ndarray
    is_red: bool

    rank: Optional[str] = None
    suit: Optional[str] = None
    rank_match: Optional[Match] = None
    suit_match: Optional[Match] = None

    @property
    def x(self) -> int:
        return min(self.rank_mark.x, self.suit_mark.x)

    @property
    def y(self) -> int:
        return self.rank_mark.y

    @property
    def identified(self) -> bool:
        return self.rank is not None and self.suit is not None

    @property
    def label(self) -> str:
        return f"{self.rank}{self.suit}" if self.identified else "??"

    @property
    def confidence(self) -> float:
        scores = [m.score for m in (self.rank_match, self.suit_match) if m]
        return min(scores) if scores else 0.0


def card_face_mask(bgr: np.ndarray) -> np.ndarray:
    """Mask over ljusa kortytor (vitt kort, oavsett filtens farg)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    light = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 150)
    mask = light.astype(np.uint8)
    # Tat ihop ytan sa glyferna raknas som "inne pa kortet"
    kernel = np.ones((max(3, bgr.shape[0] // 40) | 1,) * 2, np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def ink_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(morkt black, rott black) — det som kan vara en glyf pa ett kort."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    dark = (val < 145) & (sat < 110)
    red = (((hue < 12) | (hue > 168)) & (sat > 90) & (val > 70))
    return dark.astype(np.uint8), red.astype(np.uint8)


MIN_FACE_TO_GLYPH = 14.0
"""Ett kort ar mycket storre an sitt eget hornindex.

Chipsstaplar, namnskyltar och pott-text ar ocksa ljusa och innehaller ocksa
morka flackar — men de ljusa ytorna ar sma. Kvoten mellan ytan och glyfen
skiljer ett riktigt kort fran allt annat ljust pa bordet.
"""


def find_marks(bgr: np.ndarray, min_h: int, max_h: int) -> List[Mark]:
    """Alla blackflackar pa kortytor inom rimligt storleksspann."""
    faces = card_face_mask(bgr)
    dark, red = ink_masks(bgr)
    ink = ((dark | red) & faces).astype(np.uint8)

    # Ytorna glyferna sitter pa — inklusive glyferna sjalva, sa en kortyta
    # inte gar sonder av sitt eget tryck
    face_filled = (faces | ink).astype(np.uint8)
    fn, face_labels, face_stats, _ = cv2.connectedComponentsWithStats(face_filled, 8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    marks: List[Mark] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h or h > max_h:
            continue
        if area < 12:
            continue
        component = labels[y:y + h, x:x + w] == i

        # Vilken ljus yta ligger glyfen pa?
        face_ids, counts = np.unique(face_labels[y:y + h, x:x + w][component],
                                     return_counts=True)
        face_area = 0
        for fid, _count in sorted(zip(face_ids, counts), key=lambda t: -t[1]):
            if fid != 0:
                face_area = int(face_stats[fid][4])
                break
        if face_area < area * MIN_FACE_TO_GLYPH:
            continue

        red_px = int(np.count_nonzero(red[y:y + h, x:x + w] & component))
        dark_px = int(np.count_nonzero(dark[y:y + h, x:x + w] & component))
        marks.append(Mark(int(x), int(y), int(w), int(h), int(area),
                          is_red=red_px > dark_px, face_area=face_area))
    return marks


def pair_marks(marks: List[Mark]) -> List[Tuple[Mark, Mark]]:
    """Para ihop rankglyf med fargglyfen strax under.

    Geometrin ar det som gor detta palitligt: en rank och dess farg sitter
    tatt ihop, nastan lodratt over varandra, och ar ungefar lika stora.
    Chipsstaplar, insatsbrickor och namnskyltar uppfyller inte det.
    """
    used = set()
    pairs: List[Tuple[Mark, Mark]] = []

    # Rankkandidater: inte extremt breda eller smala. Overgransen maste
    # rymma tvasiffriga '10' — den glyfen ar klart bredare an alla andra
    # (16x11 px i 247-spelet ger aspekt 1.45+).
    ranks = [m for m in marks if 0.22 <= m.aspect <= 1.85]
    ranks.sort(key=lambda m: (m.y, m.x))

    for rank in ranks:
        if id(rank) in used:
            continue
        best = None
        best_gap = None
        for suit in marks:
            if suit is rank or id(suit) in used:
                continue
            # Fargen ska ligga under ranken
            if suit.y < rank.y + rank.h * 0.45:
                continue
            gap = suit.y - rank.bottom
            if gap > rank.h * 0.85:
                continue
            # Ungefar lodratt justerade
            if abs(suit.cx - rank.cx) > max(rank.w, suit.w) * 0.85:
                continue
            # Ungefar lika stora
            if not (0.40 <= suit.h / rank.h <= 1.70):
                continue
            # En fargsymbol ar ungefar kvadratisk — inte en textrad
            if not (0.45 <= suit.aspect <= 1.90):
                continue
            # Rank och farg hor till samma kort, alltsa samma kortyta
            if rank.face_area != suit.face_area:
                continue
            if best is None or gap < best_gap:
                best, best_gap = suit, gap
        if best is not None:
            used.add(id(rank))
            used.add(id(best))
            pairs.append((rank, best))

    return pairs


def _pad_crop(bgr: np.ndarray, m: Mark, pad: int = 2) -> np.ndarray:
    y0 = max(0, m.y - pad)
    x0 = max(0, m.x - pad)
    y1 = min(bgr.shape[0], m.bottom + pad)
    x1 = min(bgr.shape[1], m.right + pad)
    return bgr[y0:y1, x0:x1]


def find_card_candidates(
    bgr: np.ndarray,
    min_glyph_h: int = 6,
    max_glyph_frac: float = 0.30,
) -> List[CardCandidate]:
    """Hitta alla hornindex i bilden, oklassificerade."""
    max_h = max(min_glyph_h + 1, int(bgr.shape[0] * max_glyph_frac))
    marks = find_marks(bgr, min_glyph_h, max_h)

    candidates: List[CardCandidate] = []
    for rank_mark, suit_mark in pair_marks(marks):
        rank_glyph = normalize(_pad_crop(bgr, rank_mark))
        suit_glyph = normalize(_pad_crop(bgr, suit_mark))
        if rank_glyph is None or suit_glyph is None:
            continue
        candidates.append(CardCandidate(
            rank_mark=rank_mark,
            suit_mark=suit_mark,
            rank_glyph=rank_glyph,
            suit_glyph=suit_glyph,
            is_red=suit_mark.is_red or rank_mark.is_red,
        ))

    candidates.sort(key=lambda c: c.x)
    return candidates


def classify(candidates: List[CardCandidate], store: TemplateStore) -> List[CardCandidate]:
    """Satt rank/farg pa kandidaterna — men bara nar matchningen ar trovardig.

    En kandidat som inte kan avgoras lamnas medvetet oidentifierad. Ett
    tomrum i lasningen ar hanterbart; ett pahittat kort ar det inte.
    """
    for cand in candidates:
        rank_match = store.match_rank(cand.rank_glyph)
        suit_match = store.match_suit(cand.suit_glyph, is_red=cand.is_red)
        cand.rank_match = rank_match
        cand.suit_match = suit_match
        cand.rank = rank_match.label if rank_match and rank_match.trusted else None
        cand.suit = suit_match.label if suit_match and suit_match.trusted else None
    return candidates


def read_cards(bgr: np.ndarray, store: TemplateStore) -> List[CardCandidate]:
    """Fullt genomlop: hitta hornindex och klassificera dem."""
    return classify(find_card_candidates(bgr), store)


def annotate(bgr: np.ndarray, candidates: List[CardCandidate]) -> np.ndarray:
    """Rita ut vad som hittades — for kalibrering och felsokning."""
    out = bgr.copy()
    for cand in candidates:
        colour = (0, 200, 0) if cand.identified else (0, 140, 255)
        x0 = min(cand.rank_mark.x, cand.suit_mark.x) - 3
        y0 = cand.rank_mark.y - 3
        x1 = max(cand.rank_mark.right, cand.suit_mark.right) + 3
        y1 = cand.suit_mark.bottom + 3
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
        text = cand.label if cand.identified else "?"
        cv2.putText(out, text, (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out
