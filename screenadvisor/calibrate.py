"""Kalibrering — programmet far lara sig hur just den har sajtens kort ser ut.

Det ar det som gor verktyget sajtoberoende. Koden kan ingenting om nagon
specifik pokersida; den kan bara hitta hornindex. Vad en fyra eller en klover
ser ut *pa den sajt du spelar pa* lar den sig har, en gang, och sen ar
matchningen exakt istallet for gissad.

Du behover inte mata in alla 52 korten — bara 13 rankar och 4 farger, alltsa 17
glyfer. De dyker upp av sig sjalva medan du spelar nagra hander.
"""

import os
from typing import List, Optional

import cv2
import numpy as np

from screenadvisor import capture
from screenadvisor.detect import CardCandidate, annotate, classify, find_card_candidates
from screenadvisor.glyphs import RANKS, SUITS, TemplateStore
from screenadvisor.profile import Profile

SUIT_NAMES = {"s": "spader", "h": "hjarter", "d": "ruter", "c": "klover"}
WINDOW = "Kalibrering — vilket kort ar detta?"


def parse_card(text: str) -> Optional[str]:
    """Tolka '10h', 'Th', 'KH', 'th' -> 'Th'. None om ogiltigt."""
    text = (text or "").strip().replace(" ", "")
    if not text:
        return None
    if text.lower().startswith("10"):
        text = "T" + text[2:]
    if len(text) != 2:
        return None
    rank, suit = text[0].upper(), text[1].lower()
    if rank not in RANKS or suit not in SUITS:
        return None
    return rank + suit


def _show_candidate(frame: np.ndarray, cand: CardCandidate) -> None:
    """Visa kandidaten stor och i sitt sammanhang."""
    pad = 6
    y0 = max(0, cand.rank_mark.y - pad)
    y1 = min(frame.shape[0], cand.suit_mark.bottom + pad)
    x0 = max(0, min(cand.rank_mark.x, cand.suit_mark.x) - pad)
    x1 = min(frame.shape[1], max(cand.rank_mark.right, cand.suit_mark.right) + pad)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return

    scale = max(1, int(220 / max(1, crop.shape[0])))
    zoom = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    context = frame.copy()
    cv2.rectangle(context, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (0, 165, 255), 2)
    ctx_scale = min(1.0, 620 / max(1, context.shape[1]))
    if ctx_scale < 1.0:
        context = cv2.resize(context, None, fx=ctx_scale, fy=ctx_scale,
                             interpolation=cv2.INTER_AREA)

    # Satt ihop glyfen och sammanhanget i ett fonster
    height = max(zoom.shape[0], context.shape[0])
    canvas = np.zeros((height, zoom.shape[1] + context.shape[1] + 16, 3), np.uint8)
    canvas[:zoom.shape[0], :zoom.shape[1]] = zoom
    canvas[:context.shape[0], zoom.shape[1] + 16:] = context
    cv2.imshow(WINDOW, canvas)
    cv2.waitKey(1)


def missing_summary(store: TemplateStore) -> str:
    ranks, suits = store.missing()
    parts = []
    if ranks:
        parts.append("rankar kvar: " + " ".join(ranks))
    if suits:
        parts.append("farger kvar: " + " ".join(SUIT_NAMES[s] for s in suits))
    return " | ".join(parts) if parts else "alla glyfer inlarda"


def teach_from_frame(profile: Profile, frame: np.ndarray, only_unknown: bool = True) -> int:
    """Ga igenom kandidaterna i en bildruta och lat anvandaren namnge dem.

    Returnerar antalet nya glyfer som lardes in.
    """
    cands = classify(find_card_candidates(frame), profile.templates)
    if only_unknown:
        cands = [c for c in cands if not c.identified]
    if not cands:
        return 0

    learned = 0
    for cand in cands:
        _show_candidate(frame, cand)
        print()
        print("   Vad ar detta? (t.ex. Kh, 10s, 7c)")
        print("   Enter = inte ett kort   |   s = hoppa over resten   |   q = avsluta")
        try:
            answer = input("   > ").strip()
        except (EOFError, KeyboardInterrupt):
            return learned

        if answer.lower() == "q":
            raise KeyboardInterrupt
        if answer.lower() == "s":
            return learned
        if not answer:
            continue

        card = parse_card(answer)
        if card is None:
            print("   Forstod inte — hoppar over.")
            continue

        rank, suit = card[0], card[1]
        # Fargen pa glyfen maste stamma med kortets farg, annars har vi
        # sannolikt pekat pa fel sak
        expect_red = suit in ("h", "d")
        if cand.is_red != expect_red:
            sett = "rod" if cand.is_red else "svart"
            want = "rod" if expect_red else "svart"
            print(f"   Varning: glyfen ser {sett} ut men {card} ar {want}. "
                  "Sparas anda — men kontrollera att du tittade pa ratt kort.")

        profile.templates.add_rank(rank, cand.rank_glyph)
        profile.templates.add_suit(suit, cand.suit_glyph)
        learned += 1
        print(f"   Sparat: {card}   ({missing_summary(profile.templates)})")

    return learned


def calibrate_live(profile: Profile) -> None:
    """Interaktiv kalibrering mot skarmen."""
    if profile.region is None:
        print()
        print(" Valj var pokerbordet ar pa skarmen.")
        print(" Ett fonster oppnas med en bild av skarmen — dra en ruta runt bordet,")
        print(" tryck Enter. Ta med hela bordet men inte mer an nodvandigt.")
        input(" Tryck Enter for att fortsatta...")
        region = capture.select_region()
        if region is None:
            print(" Avbrutet — ingen region vald.")
            return
        profile.region = region
        profile.save()
        print(f" Region sparad: {region}")

    print()
    print(" Nu lar vi programmet kanna igen korten.")
    print(" Spela nagra hander i spelet. Nar du har nya kort framme, tryck Enter")
    print(" har — da tittar programmet pa skarmen och fragar om det den inte kanner.")
    print(f" Status: {missing_summary(profile.templates)}")

    try:
        while True:
            print()
            cmd = input(" Enter = las skarmen nu  |  a = las aven kanda kort  |  q = klar > ").strip().lower()
            if cmd == "q":
                break
            frame = capture.grab(profile.region)
            learned = teach_from_frame(profile, frame, only_unknown=(cmd != "a"))
            if learned:
                profile.save()
            else:
                print("   Inga nya glyfer att lara in i den har bilden.")
            print(f"   {missing_summary(profile.templates)}")
            if profile.templates.is_complete():
                print()
                print("   Alla 13 rankar och 4 farger ar inlarda — kalibreringen ar klar.")
                break
    except KeyboardInterrupt:
        print()
    finally:
        cv2.destroyAllWindows()
        profile.save()
        print(f" Profil sparad: {profile.path}")


def calibrate_from_images(profile: Profile, paths: List[str]) -> None:
    """Kalibrera fran sparade skarmdumpar istallet for direkt fran skarmen."""
    try:
        for path in paths:
            if not os.path.exists(path):
                print(f" Hittar inte {path}")
                continue
            data = np.fromfile(path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                print(f" Kunde inte lasa {path}")
                continue
            print()
            print(f" === {os.path.basename(path)}")
            teach_from_frame(profile, frame, only_unknown=True)
            profile.save()
            print(f"   {missing_summary(profile.templates)}")
    except KeyboardInterrupt:
        print()
    finally:
        cv2.destroyAllWindows()
        profile.save()
        print(f" Profil sparad: {profile.path}")
