"""
Card Detection Diagnostic Tool

Captures current PokerStars frame, extracts every card region,
runs both template matching and OCR, and saves annotated debug images
so you can see exactly what the system detects vs what's actually there.

Usage: python debug_card_detection.py
"""

import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
os.chdir(_dir)

import cv2
import numpy as np
import mss

from calibrate_pokerstars import find_pokerstars_window, calculate_regions
from vision.table_reader import TableReader, TableRegions, RANKS, SUITS

DEBUG_DIR = os.path.join(_dir, "debug_images")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_image(path, img):
    success, buf = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())


def capture_window():
    """Capture the PokerStars window."""
    ps = find_pokerstars_window()
    if not ps:
        print("PokerStars not found!")
        return None, None

    title, wx, wy, ww, wh = ps
    wx, wy = max(0, wx), max(0, wy)
    print(f"Found: {title}")
    print(f"  Window: {ww}x{wh} at ({wx},{wy})")

    with mss.mss() as sct:
        region = {"top": wy, "left": wx, "width": ww, "height": wh}
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame, (ww, wh)


def main():
    print("=" * 55)
    print("  CARD DETECTION DIAGNOSTIC")
    print("=" * 55)
    print()

    # Capture frame
    frame, (ww, wh) = capture_window() or (None, None)
    if frame is None:
        return

    # Save full frame
    save_image(os.path.join(DEBUG_DIR, "diag_full_frame.png"), frame)
    print(f"  Full frame saved: {frame.shape}")

    # Calculate regions (relative to window at 0,0)
    calc = calculate_regions(0, 0, ww, wh)
    regions = TableRegions()
    regions.hero_card_1 = tuple(calc["hero_card_1"])
    regions.hero_card_2 = tuple(calc["hero_card_2"])
    regions.community_cards = [tuple(c) for c in calc["community_cards"]]
    regions.pot_text = tuple(calc["pot_text"])
    regions.player_regions = calc["player_regions"]

    # Initialize table reader
    template_dir = os.path.join(_dir, "models", "card_templates")
    reader = TableReader(
        template_dir=template_dir,
        regions=regions,
        tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    print(f"\n{'='*55}")
    print("  HERO CARDS")
    print(f"{'='*55}")

    hero_regions = [
        ("Hero Card 1", regions.hero_card_1),
        ("Hero Card 2", regions.hero_card_2),
    ]

    annotated = frame.copy()

    for label, region in hero_regions:
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            print(f"\n{label}: EMPTY ROI at {region}")
            continue

        print(f"\n{label} region: x={x} y={y} w={w} h={h}")
        print(f"  ROI shape: {roi.shape}")

        # Save the raw ROI
        safe_label = label.replace(" ", "_")
        save_image(os.path.join(DEBUG_DIR, f"diag_{safe_label}_roi.png"), roi)

        # Check if card present
        present = reader._is_card_present(roi)
        print(f"  Card present: {present}")

        if not present:
            print(f"  SKIPPED (no card detected)")
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(annotated, f"{label}: NO CARD", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            continue

        # Template matching
        tmpl_result = reader._detect_card(roi)
        print(f"  Template match: {tmpl_result}")

        # OCR
        ocr_result = reader._detect_card_ocr(roi)
        print(f"  OCR result:     {ocr_result}")

        # Suit color analysis
        h_roi, w_roi = roi.shape[:2]
        suit_area = roi[int(h_roi * 0.25):int(h_roi * 0.55), 0:int(w_roi * 0.40)]
        if suit_area.size > 0:
            hsv = cv2.cvtColor(suit_area, cv2.COLOR_BGR2HSV)
            total = suit_area.shape[0] * suit_area.shape[1]

            red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
            red2 = cv2.inRange(hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
            red_px = cv2.countNonZero(red1) + cv2.countNonZero(red2)
            blue = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([130, 255, 255]))
            blue_px = cv2.countNonZero(blue)
            green = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
            green_px = cv2.countNonZero(green)

            print(f"  Suit colors: red={red_px/total*100:.1f}% blue={blue_px/total*100:.1f}% green={green_px/total*100:.1f}%")

            # Save suit area
            save_image(os.path.join(DEBUG_DIR, f"diag_{safe_label}_suit_area.png"), suit_area)

            # Also save HSV channels for debugging
            h_ch, s_ch, v_ch = cv2.split(hsv)
            save_image(os.path.join(DEBUG_DIR, f"diag_{safe_label}_hue.png"), h_ch)
            save_image(os.path.join(DEBUG_DIR, f"diag_{safe_label}_sat.png"), s_ch)

        # Draw on annotated image
        color = (0, 255, 0) if tmpl_result or ocr_result else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        detected = tmpl_result or ocr_result or "???"
        cv2.putText(annotated, f"{label}: {detected}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Corner matching details — show what the corner looks like
        rh, rw = roi.shape[:2]
        corner = roi[0:int(rh * 0.45), 0:int(rw * 0.45)]
        if corner.size > 0:
            # Scale up for visibility
            corner_big = cv2.resize(corner, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            save_image(os.path.join(DEBUG_DIR, f"diag_{safe_label}_corner_4x.png"), corner_big)

            # Also show what OCR sees (grayscale + threshold)
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            gray_big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray_big, 140, 255, cv2.THRESH_BINARY)
            save_image(os.path.join(DEBUG_DIR, f"diag_{safe_label}_corner_thresh.png"), thresh)

        # Template scores — show top 5 matches
        if hasattr(reader, '_corner_cache') and reader._corner_cache:
            roi_corner = roi[0:int(rh * 0.45), 0:int(rw * 0.45)]
            scores = []
            for card_name, corner_tmpl in reader._corner_cache.items():
                ch, cw = corner_tmpl.shape[:2]
                rch, rcw = roi_corner.shape[:2]
                if ch != rch or cw != rcw:
                    if ch > rch or cw > rcw:
                        scale = min(rch / ch, rcw / cw)
                        tmpl = cv2.resize(corner_tmpl, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA)
                    else:
                        tmpl = corner_tmpl
                else:
                    tmpl = corner_tmpl
                if tmpl.shape[0] > rch or tmpl.shape[1] > rcw:
                    continue
                result = cv2.matchTemplate(roi_corner, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                scores.append((card_name, max_val))

            scores.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 template scores:")
            for name, score in scores[:5]:
                marker = " <-- BEST" if name == scores[0][0] else ""
                print(f"    {name}: {score:.4f}{marker}")

    print(f"\n{'='*55}")
    print("  COMMUNITY CARDS")
    print(f"{'='*55}")

    for i, region in enumerate(regions.community_cards):
        x, y, w, h = region
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            print(f"\nCC{i+1}: EMPTY ROI")
            continue

        label = f"CC{i+1}"
        print(f"\n{label} region: x={x} y={y} w={w} h={h}")

        save_image(os.path.join(DEBUG_DIR, f"diag_{label}_roi.png"), roi)

        present = reader._is_card_present(roi)
        print(f"  Card present: {present}")

        if not present:
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (128, 128, 128), 1)
            continue

        tmpl_result = reader._detect_card(roi)
        ocr_result = reader._detect_card_ocr(roi)
        print(f"  Template match: {tmpl_result}")
        print(f"  OCR result:     {ocr_result}")

        # Suit colors
        h_roi, w_roi = roi.shape[:2]
        suit_area = roi[int(h_roi * 0.25):int(h_roi * 0.55), 0:int(w_roi * 0.40)]
        if suit_area.size > 0:
            hsv = cv2.cvtColor(suit_area, cv2.COLOR_BGR2HSV)
            total = suit_area.shape[0] * suit_area.shape[1]
            red1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([12, 255, 255]))
            red2 = cv2.inRange(hsv, np.array([168, 80, 80]), np.array([180, 255, 255]))
            red_px = cv2.countNonZero(red1) + cv2.countNonZero(red2)
            blue = cv2.inRange(hsv, np.array([95, 50, 50]), np.array([130, 255, 255]))
            blue_px = cv2.countNonZero(blue)
            green = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
            green_px = cv2.countNonZero(green)
            print(f"  Suit colors: red={red_px/total*100:.1f}% blue={blue_px/total*100:.1f}% green={green_px/total*100:.1f}%")

        # Save corner
        rh, rw = roi.shape[:2]
        corner = roi[0:int(rh * 0.45), 0:int(rw * 0.45)]
        if corner.size > 0:
            corner_big = cv2.resize(corner, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
            save_image(os.path.join(DEBUG_DIR, f"diag_{label}_corner_4x.png"), corner_big)

        color = (0, 255, 0) if tmpl_result or ocr_result else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        detected = tmpl_result or ocr_result or "---"
        cv2.putText(annotated, detected, (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Save annotated full frame
    save_image(os.path.join(DEBUG_DIR, "diag_annotated.png"), annotated)

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"\nAll images saved to: {DEBUG_DIR}/diag_*")
    print("Files:")
    print("  diag_full_frame.png       — Full captured frame")
    print("  diag_annotated.png        — Frame with detection results")
    print("  diag_Hero_Card_*_roi.png  — Raw card regions")
    print("  diag_Hero_Card_*_corner_4x.png — Corner at 4x zoom")
    print("  diag_Hero_Card_*_corner_thresh.png — Thresholded for OCR")
    print("  diag_Hero_Card_*_suit_area.png — Suit detection area")
    print("  diag_CC*_roi.png          — Community card regions")
    print()
    print("Compare the ROI images with what you see on PokerStars!")
    print("If the ROI shows the wrong area, the calibration is off.")
    print("If the ROI is correct but detection is wrong, it's a matching issue.")


if __name__ == "__main__":
    main()
