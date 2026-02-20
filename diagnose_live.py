"""
Diagnostic tool — captures ONE frame from PokerStars and tests every
detection step. Saves debug images for each region so we can see exactly
what's going wrong.

Usage: python diagnose_live.py
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import mss

_dir = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(_dir, "debug_images", "diagnose")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_img(name, img):
    path = os.path.join(DEBUG_DIR, name)
    ok, buf = cv2.imencode('.png', img)
    if ok:
        with open(path, 'wb') as f:
            f.write(buf)


def main():
    from calibrate_pokerstars import find_pokerstars_window, calculate_regions
    from vision.table_reader import TableReader, TableRegions, RANKS, SUITS
    from config import Config

    # 1. Find window WITHOUT bringing to front (avoids Windows Snap resizing)
    ps = find_pokerstars_window(bring_to_front=False)
    if not ps:
        print("PokerStars window not found!")
        return
    title, wx, wy, ww, wh = ps
    print(f"Window: \"{title}\"")
    print(f"  Size: {ww}x{wh} at ({wx},{wy})")

    import time
    time.sleep(0.3)  # Brief pause for stable capture

    # 2. Capture frame
    with mss.mss() as sct:
        region = {"top": wy, "left": wx, "width": ww, "height": wh}
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    save_img("00_full_frame.png", frame)
    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    # 3. Auto-detect layout from card positions
    from calibrate_pokerstars import auto_detect_layout
    calc = auto_detect_layout(frame)
    if calc is None:
        print("\nINGA KORT SYNLIGA — kan inte auto-detektera layout!")
        print("Starta ett spel med kort pa bordet och kor igen.")
        calc = calculate_regions(0, 0, ww, wh)
        print("Anvander grova fallback-proportioner.\n")

    regions = TableRegions()
    regions.hero_card_1 = tuple(calc["hero_card_1"])
    regions.hero_card_2 = tuple(calc["hero_card_2"])
    regions.community_cards = [tuple(c) for c in calc["community_cards"]]
    regions.pot_text = tuple(calc["pot_text"])
    regions.player_regions = calc["player_regions"]

    # 4. Draw ALL regions on frame
    overlay = frame.copy()
    for label, reg in [("HERO1", regions.hero_card_1), ("HERO2", regions.hero_card_2)]:
        x, y, rw, rh = reg
        cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 255, 0), 2)
        cv2.putText(overlay, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for i, reg in enumerate(regions.community_cards):
        x, y, rw, rh = reg
        cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (255, 255, 0), 2)
        cv2.putText(overlay, f"COMM{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    x, y, rw, rh = regions.pot_text
    cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 255, 255), 2)
    cv2.putText(overlay, "POT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    save_img("01_overlay.png", overlay)

    # 5. Load templates and init reader
    config = Config()
    reader = TableReader(
        template_dir=config.template_dir,
        regions=regions,
        tesseract_cmd=config.tesseract_cmd,
    )

    # 6. Test each hero card region
    print("\n=== HERO CARDS ===")
    for idx, reg in enumerate([regions.hero_card_1, regions.hero_card_2]):
        x, y, rw, rh = reg
        roi = frame[y:y+rh, x:x+rw]
        save_img(f"hero_{idx}_roi.png", roi)

        present = reader._is_card_present(roi)
        print(f"  Hero {idx+1}: region=({x},{y},{rw},{rh}), present={present}")

        if present:
            # Template match
            tmpl_result = reader._detect_card(roi)
            print(f"    Template match: {tmpl_result}")

            # OCR
            ocr_result = reader._detect_card_ocr(roi)
            print(f"    OCR result: {ocr_result}")

            # Suit detection with detailed debug
            suit = reader._detect_suit(roi)
            print(f"    Suit detected: {suit}")

            # Save the actual suit crop used by _detect_suit
            h, w = roi.shape[:2]
            suit_crop = roi[int(h*0.25):int(h*0.45), 0:int(w*0.38)]
            save_img(f"hero_{idx}_suit_crop.png", suit_crop)

            # Analyze the suit crop
            if suit_crop.size > 0:
                sc_hsv = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2HSV)
                sc_total = suit_crop.shape[0] * suit_crop.shape[1]
                r1 = cv2.inRange(sc_hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
                r2 = cv2.inRange(sc_hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
                red_mask = cv2.bitwise_or(r1, r2)
                red_r = cv2.countNonZero(red_mask) / max(sc_total, 1)
                print(f"    Suit crop: {suit_crop.shape[1]}x{suit_crop.shape[0]}, red_ratio={red_r:.3f}")

                # Show contour analysis if red
                if red_r > 0.03:
                    kernel = np.ones((2, 2), np.uint8)
                    red_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
                    red_big = cv2.resize(red_closed, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(red_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    save_img(f"hero_{idx}_red_mask.png", red_big)
                    for ci, cnt in enumerate(sorted(cnts, key=cv2.contourArea, reverse=True)[:3]):
                        ca = cv2.contourArea(cnt)
                        if ca < 10:
                            continue
                        hull = cv2.convexHull(cnt)
                        sol = ca / max(cv2.contourArea(hull), 1)
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        asp = bw / max(bh, 1)
                        eps = 0.04 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, eps, True)
                        print(f"    Contour {ci}: area={ca:.0f}, solidity={sol:.3f}, "
                              f"aspect={asp:.2f}, vertices={len(approx)}, bbox=({bx},{by},{bw},{bh})")

            # Image stats
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            print(f"    ROI stats: mean={gray.mean():.1f}, std={gray.std():.1f}, bright_pct={np.sum(gray>150)/max(h*w,1):.3f}")
        else:
            print(f"    (not present - skipped)")

    # 7. Test each community card region
    print("\n=== COMMUNITY CARDS ===")
    for idx, reg in enumerate(regions.community_cards):
        x, y, rw, rh = reg
        roi = frame[y:y+rh, x:x+rw]
        save_img(f"comm_{idx}_roi.png", roi)

        present = reader._is_card_present(roi)
        print(f"  Comm {idx+1}: region=({x},{y},{rw},{rh}), size={rw}x{rh}, present={present}")

        if present:
            tmpl_result = reader._detect_card(roi)
            ocr_result = reader._detect_card_ocr(roi)
            suit = reader._detect_suit(roi)
            print(f"    Template: {tmpl_result}, OCR: {ocr_result}, Suit: {suit}")

            # Corner used for matching
            corner_h = int(rh * 0.45)
            corner_w = int(rw * 0.45)
            corner = roi[0:corner_h, 0:corner_w]
            save_img(f"comm_{idx}_corner.png", corner)
            print(f"    Corner size: {corner_w}x{corner_h}")

            # Top 3 template matches
            rch, rcw = corner.shape[:2]
            scores = []
            for card_name, corner_tmpl in reader._corner_cache.items():
                tmpl = cv2.resize(corner_tmpl, (rcw, rch), interpolation=cv2.INTER_AREA)
                result = cv2.matchTemplate(corner, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                scores.append((max_val, card_name))
            scores.sort(reverse=True)
            print(f"    Top 5: {[(f'{s:.3f}', n) for s, n in scores[:5]]}")
        else:
            # Show why it was rejected
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 220]))
            green_r = cv2.countNonZero(green) / max(roi.shape[0] * roi.shape[1], 1)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            print(f"    REJECTED: green_ratio={green_r:.3f}, gray_mean={gray.mean():.1f}")

    # 8. Test pot region
    print("\n=== POT ===")
    x, y, rw, rh = regions.pot_text
    pot_roi = frame[y:y+rh, x:x+rw]
    save_img("pot_roi.png", pot_roi)
    pot_text = reader._read_text(frame, regions.pot_text, mode="amount")
    pot_val = reader._parse_amount(pot_text)
    print(f"  Region: ({x},{y},{rw},{rh})")
    print(f"  OCR text: '{pot_text}'")
    print(f"  Parsed: {pot_val}")

    # 9. Search for white card rectangles on the table to find true card positions
    print("\n=== CARD POSITION DETECTION ===")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Cards are bright white rectangles on the green felt
    _, white_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    # Focus on center area where community cards are (y: 25-55%, x: 25-85%)
    center_mask = np.zeros_like(white_mask)
    y1, y2 = int(wh * 0.25), int(wh * 0.55)
    x1, x2 = int(ww * 0.25), int(ww * 0.85)
    center_mask[y1:y2, x1:x2] = white_mask[y1:y2, x1:x2]

    contours, _ = cv2.findContours(center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_rects = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        aspect = cw / max(ch, 1)
        # Card should be roughly 50-120px wide, 70-160px tall, aspect 0.5-0.9
        if 40 < cw < 150 and 60 < ch < 200 and 0.4 < aspect < 1.0 and area > 2000:
            card_rects.append((x, y, cw, ch))

    card_rects.sort(key=lambda r: r[0])  # Sort left to right

    card_overlay = frame.copy()
    for i, (x, y, cw, ch) in enumerate(card_rects):
        cv2.rectangle(card_overlay, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
        cv2.putText(card_overlay, f"CARD{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        pct_x = x / ww
        pct_y = y / wh
        pct_w = cw / ww
        pct_h = ch / wh
        print(f"  Card {i}: pixel=({x},{y},{cw},{ch}), pct=({pct_x:.3f},{pct_y:.3f},{pct_w:.3f},{pct_h:.3f})")
    save_img("10_card_positions.png", card_overlay)

    if card_rects:
        # Calculate community card layout from detected positions
        # First card position and gap
        first_x = card_rects[0][0] / ww
        first_y = card_rects[0][1] / wh
        avg_w = np.mean([r[2] for r in card_rects]) / ww
        avg_h = np.mean([r[3] for r in card_rects]) / wh
        if len(card_rects) > 1:
            gaps = [(card_rects[i+1][0] - card_rects[i][0]) / ww for i in range(len(card_rects)-1)]
            avg_gap = np.mean(gaps)
        else:
            avg_gap = 0.066
        print(f"\n  Suggested PS_LAYOUT community:")
        print(f"    start_x: {first_x:.3f}")
        print(f"    y: {first_y:.3f}")
        print(f"    w: {avg_w:.3f}")
        print(f"    h: {avg_h:.3f}")
        print(f"    gap: {avg_gap:.3f}")

    # 10. Search for pot text area
    print("\n=== POT TEXT SEARCH ===")
    # Look for white/yellow text above the community cards
    # Pot text is typically in the top-center, above the cards
    pot_search_y1 = int(wh * 0.20)
    pot_search_y2 = int(wh * 0.40)
    pot_search_x1 = int(ww * 0.40)
    pot_search_x2 = int(ww * 0.75)
    pot_area = frame[pot_search_y1:pot_search_y2, pot_search_x1:pot_search_x2]
    save_img("11_pot_search_area.png", pot_area)
    try:
        import pytesseract as pt
        if config.tesseract_cmd:
            pt.pytesseract.tesseract_cmd = config.tesseract_cmd
        pot_gray = cv2.cvtColor(pot_area, cv2.COLOR_BGR2GRAY)
        _, pot_thresh = cv2.threshold(pot_gray, 140, 255, cv2.THRESH_BINARY)
        pot_scaled = cv2.resize(pot_thresh, None, fx=2, fy=2)
        save_img("11_pot_thresh.png", pot_scaled)
        text = pt.image_to_string(pot_scaled, config="--psm 6").strip()
        print(f"  Text found in pot area: '{text}'")
    except ImportError:
        print("  pytesseract not available")

    # Also look for hero card positions (bottom center area)
    print("\n=== HERO CARD SEARCH ===")
    hero_mask = np.zeros_like(white_mask)
    hy1, hy2 = int(wh * 0.55), int(wh * 0.85)
    hx1, hx2 = int(ww * 0.30), int(ww * 0.70)
    hero_mask[hy1:hy2, hx1:hx2] = white_mask[hy1:hy2, hx1:hx2]
    h_contours, _ = cv2.findContours(hero_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hero_cards = []
    for c in h_contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        aspect = cw / max(ch, 1)
        if 30 < cw < 150 and 50 < ch < 200 and 0.4 < aspect < 1.0 and area > 1500:
            hero_cards.append((x, y, cw, ch))
    hero_cards.sort(key=lambda r: r[0])
    for i, (x, y, cw, ch) in enumerate(hero_cards):
        pct_x = x / ww
        pct_y = y / wh
        pct_w = cw / ww
        pct_h = ch / wh
        print(f"  Hero card {i}: pixel=({x},{y},{cw},{ch}), pct=({pct_x:.3f},{pct_y:.3f},{pct_w:.3f},{pct_h:.3f})")
        cv2.rectangle(card_overlay, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
    save_img("10_card_positions.png", card_overlay)

    # 11. Full read_table
    print("\n=== FULL READ ===")
    reading = reader.read_table(frame)
    print(f"  Hero: {reading.hero_cards}")
    print(f"  Community: {reading.community_cards}")
    print(f"  Pot: {reading.pot}")
    print(f"  Players: {len(reading.players)}")
    print(f"  Dealer: {reading.dealer_seat}")

    print(f"\nAll debug images saved in: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
