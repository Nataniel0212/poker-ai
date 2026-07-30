"""Satter ihop en lasning av bordet fran en enskild bildruta.

En trevlig egenskap hos de har spelen: motstandarnas kort ligger med ryggen upp
och har darfor inget hornindex. Kortlasaren hittar alltsa bara de kort som
faktiskt ar synliga — dina egna och bordets. Motstandarna raknas separat genom
att leta efter kortryggar.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from screenadvisor.detect import CardCandidate, classify, find_card_candidates
from screenadvisor.glyphs import TemplateStore


@dataclass
class TableRead:
    hero: List[str] = field(default_factory=list)
    board: List[str] = field(default_factory=list)
    opponents: int = 0
    unknown_cards: int = 0
    candidates: List[CardCandidate] = field(default_factory=list)
    note: str = ""

    @property
    def usable(self) -> bool:
        """Kan vi ge rad? Kraver tva hjaltekort och inga oklara kort."""
        return len(self.hero) == 2 and self.unknown_cards == 0

    @property
    def street(self) -> str:
        n = len(self.board)
        if n == 0:
            return "preflop"
        if n == 3:
            return "flop"
        if n == 4:
            return "turn"
        if n == 5:
            return "river"
        return "okand"

    @property
    def all_cards(self) -> List[str]:
        return self.hero + self.board


def count_card_backs(bgr: np.ndarray) -> int:
    """Uppskatta antalet motstandare genom att rakna kortryggar.

    Kortryggar ar starkt mattade enfargade ytor (bla i de flesta av dessa spel).
    Tva ryggar per spelare, men de overlappar ofta och smalter ihop till en yta,
    sa vi raknar ytor och inte kort.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Mattad, men inte gron filt (filt ~ hue 35-85) och inte rod skylt
    backs = (sat > 120) & (val > 70) & (hue > 85) & (hue < 140)
    mask = cv2.morphologyEx(backs.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    frame_area = bgr.shape[0] * bgr.shape[1]
    # En kortryggsyta ar liten men inte forsumbar
    min_area = frame_area * 0.0015
    max_area = frame_area * 0.05
    return sum(1 for i in range(1, n) if min_area <= stats[i][4] <= max_area)


def count_genuine_unknowns(
    rejected: List[CardCandidate],
    identified: List[CardCandidate],
) -> int:
    """Hur manga av de forkastade kandidaterna ar faktiskt olasta *kort*?

    Skillnaden ar viktig. Kortlasaren foreslar fler kandidater an det finns
    kort — chipsstaplar och stora fargsymboler i kortens mitt kan se ut som
    ett hornindex vid forsta anblicken. Om alla sadana raknades som "okant
    kort" skulle verktyget aldrig vaga ge nagot rad alls.

    Men motsatsen ar varre: att tiga om ett kort vi verkligen inte kunde lasa.
    Darfor jamfors kandidaten med de kort vi *kunde* lasa — ett riktigt
    hornindex sitter pa en kortstor yta och har ungefar samma glyfstorlek som
    de ovriga korten i samma lasning.
    """
    if not rejected:
        return 0
    if not identified:
        # Utan referens kan vi inte avgora nagot — da ar allt osakert
        return len(rejected)

    median_h = float(np.median([c.rank_mark.h for c in identified]))
    median_face = float(np.median([c.rank_mark.face_area for c in identified]))

    # Ytor dar vi redan last ett hornindex. En forkastad kandidat pa samma
    # kortyta ar kortets egen dekor (mittsymboler), inte ett olast kort —
    # annars larmar varje lasning av en solfjader med stora pips i onodan.
    identified_faces = {c.rank_mark.face_area for c in identified}

    unknowns = 0
    for cand in rejected:
        if cand.rank_mark.face_area in identified_faces:
            continue
        on_card = cand.rank_mark.face_area >= median_face * 0.5
        similar_size = 0.6 <= (cand.rank_mark.h / median_h) <= 1.5
        if on_card and similar_size:
            unknowns += 1
    return unknowns


def group_rows(cands: List[CardCandidate], tolerance: float) -> List[List[CardCandidate]]:
    """Gruppera kort i vagrata rader efter y-position."""
    rows: List[List[CardCandidate]] = []
    for cand in sorted(cands, key=lambda c: c.y):
        placed = False
        for row in rows:
            if abs(row[0].y - cand.y) <= tolerance:
                row.append(cand)
                placed = True
                break
        if not placed:
            rows.append([cand])
    for row in rows:
        row.sort(key=lambda c: c.x)
    return rows


def read_table(
    bgr: np.ndarray,
    store: TemplateStore,
    hero_zone: Optional[Tuple[int, int, int, int]] = None,
    opponents_override: Optional[int] = None,
) -> TableRead:
    """Las hjaltekort, bordskort och antal motstandare ur en bildruta."""
    cands = classify(find_card_candidates(bgr), store)
    identified = [c for c in cands if c.identified]
    rejected = [c for c in cands if not c.identified]
    unknown = count_genuine_unknowns(rejected, identified)

    result = TableRead(candidates=cands, unknown_cards=unknown)

    if not identified:
        result.note = "Inga kort hittade"
        return result

    # Dubblettskydd: samma kort kan inte forekomma tva ganger. Om det hander
    # har vi last fel nagonstans, och da ar det ohederligt att ge rad.
    labels = [c.label for c in identified]
    if len(set(labels)) != len(labels):
        result.note = "Dubblerade kort i lasningen — osaker"
        result.unknown_cards = max(1, unknown)
        return result

    glyph_h = np.median([c.rank_mark.h for c in identified])
    rows = group_rows(identified, tolerance=max(8.0, glyph_h * 2.2))

    hero_cards: List[CardCandidate] = []
    if hero_zone is not None:
        hx, hy, hw, hh = hero_zone
        hero_cards = [
            c for c in identified
            if hx <= c.x <= hx + hw and hy <= c.y <= hy + hh
        ]
        hero_cards.sort(key=lambda c: c.x)
    if len(hero_cards) != 2:
        # Fallback pa geometri: hjaltens kort ligger langst ner — men bara om
        # raden faktiskt ligger i nedre delen av bilden. Annars ar det bordet
        # vi rakat lasa (t.ex. nar hjaltens egna kort inte kunde lasas), och
        # att kalla bordskort for hjaltekort ger sjalvsakra rad pa fel hand.
        bottom_row = max(rows, key=lambda r: r[0].y)
        in_lower_half = bottom_row[0].y > bgr.shape[0] * 0.45
        hero_cards = bottom_row if (len(bottom_row) == 2 and in_lower_half) else []

    hero_ids = {id(c) for c in hero_cards}
    board_cards = [c for c in identified if id(c) not in hero_ids]
    if hero_zone is not None:
        # Ett kort inne i hjaltezonen ar aldrig ett bordskort. Utan den har
        # sparren blev ett ensamt last hjaltekort presenterat som bord.
        hx, hy, hw, hh = hero_zone
        board_cards = [
            c for c in board_cards
            if not (hx <= c.x <= hx + hw and hy <= c.y <= hy + hh)
        ]
    board_cards.sort(key=lambda c: c.x)

    result.hero = [c.label for c in hero_cards]
    result.board = [c.label for c in board_cards]

    # Med en hjaltezon vet vi var olasta kort *kan* paverka radet: i zonen
    # eller pa bordsraden. Ett kortlikt fynd nagon annanstans (chipshogar
    # intill motstandarnas ljusa namnskyltar, deras uppvisade kort vid
    # showdown) andrar inte radet och far inte blockera det.
    if hero_zone is not None and identified:
        median_h = float(np.median([c.rank_mark.h for c in identified]))
        median_face = float(np.median([c.rank_mark.face_area for c in identified]))
        identified_faces = {c.rank_mark.face_area for c in identified}
        board_ys = [c.y for c in board_cards]
        hx, hy, hw, hh = hero_zone

        relevant = 0
        for cand in rejected:
            if cand.rank_mark.face_area in identified_faces:
                continue
            on_card = cand.rank_mark.face_area >= median_face * 0.5
            similar = 0.6 <= (cand.rank_mark.h / median_h) <= 1.5
            if not (on_card and similar):
                continue
            in_zone = hx <= cand.x <= hx + hw and hy <= cand.y <= hy + hh
            on_board_row = any(abs(cand.y - by) <= median_h * 2.2 for by in board_ys)
            if in_zone or on_board_row:
                relevant += 1
        result.unknown_cards = relevant

    if len(board_cards) > 5:
        result.note = "Fler an fem bordskort — troligen showdown"
        result.board = []
        result.unknown_cards = max(1, unknown)
        return result

    # Antal motstandare ar medvetet en instalning och inte en lasning.
    # Kortryggar overlappar olika mycket i olika spel, sa en raknare blir
    # antingen for hog eller for lag — och equity beror markbart pa siffran.
    # Detektionen far vara ett forslag; anvandaren har sista ordet.
    if opponents_override is not None:
        result.opponents = max(1, opponents_override)
    else:
        result.opponents = max(1, count_card_backs(bgr))

    if len(hero_cards) != 2:
        result.note = "Hittade inte tva egna kort"

    return result
