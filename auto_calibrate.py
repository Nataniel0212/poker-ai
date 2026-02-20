"""
Fully automatic calibration for PokerStars.
Uses OCR to find ALL text on the table and maps it to player positions.
No clicks needed, no manual proportions.
"""

import sys
import os
import io
import json
import ctypes
from ctypes import wintypes

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    _dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _dir = os.getcwd()
sys.path.insert(0, _dir)
os.chdir(_dir)

import cv2
import numpy as np
import mss
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def find_pokerstars_window():
    """Find PokerStars window using Windows API."""
    user32 = ctypes.windll.user32
    windows = []

    def enum_handler(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if title:
                    rect = wintypes.RECT()
                    user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    x, y, x2, y2 = rect.left, rect.top, rect.right, rect.bottom
                    w, h = x2 - x, y2 - y
                    if w > 200 and h > 200:
                        windows.append((title, x, y, w, h))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_handler), 0)

    for title, x, y, w, h in windows:
        tl = title.lower()
        if ('hold' in tl or 'omaha' in tl) and ('limit' in tl or 'no limit' in tl):
            return (title, x, y, w, h)
        if 'pokerstars' in tl and w > 400:
            return (title, x, y, w, h)
    for title, x, y, w, h in windows:
        tl = title.lower()
        if any(kw in tl for kw in ['poker', 'gadolin', 'stars']):
            return (title, x, y, w, h)
    return None


def find_all_text(frame, region):
    """Find all text in a region using OCR. Returns list of {text, x, y, w, h}."""
    x0, y0, rw, rh = region
    crop = frame[y0:y0+rh, x0:x0+rw]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Try multiple thresholds for better detection
    all_texts = {}
    for threshold in [140, 160, 180]:
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_data(thresh, config='--psm 11', output_type=pytesseract.Output.DICT)

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if text and conf > 35 and len(text) > 1:
                tx, ty = data['left'][i] + x0, data['top'][i] + y0
                tw, th = data['width'][i], data['height'][i]
                key = f"{tx//10}_{ty//10}"  # Dedup nearby texts
                if key not in all_texts or conf > all_texts[key]['conf']:
                    all_texts[key] = {
                        'text': text, 'x': tx, 'y': ty, 'w': tw, 'h': th, 'conf': conf
                    }

    return list(all_texts.values())


def classify_text(text):
    """Classify text as name, number, action, or other."""
    text_clean = text.replace(' ', '').replace('.', '').replace(',', '')
    if text_clean.isdigit():
        return 'number'
    if text.lower() in ('fold', 'check', 'call', 'raise', 'bet', 'all-in',
                         'betalar', 'sb', 'bb', 'pott:', 'pott', 'dealer:'):
        return 'action'
    if len(text) >= 3 and any(c.isalpha() for c in text):
        return 'name'
    return 'other'


def find_player_regions(frame, win_x, win_y, win_w, win_h):
    """Auto-detect player name/stack positions using OCR."""

    # Effective visible area
    eff_w = min(win_w, 1920 - max(0, win_x))
    eff_h = min(win_h, 1080 - max(0, win_y))
    wx = max(0, win_x)
    wy = max(0, win_y)

    # Define search zones for each player position (wider than needed)
    # Format: (x_pct, y_pct, w_pct, h_pct) relative to effective window
    zones = {
        'hero':  (0.30, 0.64, 0.35, 0.16),
        'br':    (0.65, 0.45, 0.33, 0.20),
        'tr':    (0.65, 0.15, 0.33, 0.20),
        'tc':    (0.25, 0.05, 0.40, 0.17),
        'tl':    (0.00, 0.15, 0.30, 0.20),
        'bl':    (0.00, 0.45, 0.30, 0.20),
    }

    # Pot zone
    pot_zone = (0.35, 0.28, 0.28, 0.08)

    players = {}
    for zone_name, (zx, zy, zw, zh) in zones.items():
        region = (
            int(wx + zx * eff_w),
            int(wy + zy * eff_h),
            int(zw * eff_w),
            int(zh * eff_h),
        )
        texts = find_all_text(frame, region)

        # Find the best name and number
        names = [t for t in texts if classify_text(t['text']) == 'name']
        numbers = [t for t in texts if classify_text(t['text']) == 'number']

        # Sort names by confidence, pick best
        names.sort(key=lambda t: t['conf'], reverse=True)
        numbers.sort(key=lambda t: t['conf'], reverse=True)

        name_info = names[0] if names else None
        # Stack is the number closest to the name (vertically)
        stack_info = None
        if name_info and numbers:
            numbers.sort(key=lambda t: abs(t['y'] - name_info['y']))
            stack_info = numbers[0]
        elif numbers:
            stack_info = numbers[0]

        players[zone_name] = {'name': name_info, 'stack': stack_info}

    # Pot
    pot_region = (
        int(wx + pot_zone[0] * eff_w),
        int(wy + pot_zone[1] * eff_h),
        int(pot_zone[2] * eff_w),
        int(pot_zone[3] * eff_h),
    )
    pot_texts = find_all_text(frame, pot_region)
    pot_info = None
    for t in pot_texts:
        if 'pott' in t['text'].lower() or 'pot' in t['text'].lower():
            pot_info = t
            break
    if not pot_info:
        # Fall back to any number in pot area
        pot_numbers = [t for t in pot_texts if classify_text(t['text']) == 'number']
        if pot_numbers:
            pot_info = pot_numbers[0]

    return players, pot_info


def build_regions(players, pot_info, win_x, win_y, win_w, win_h):
    """Build the table_regions.json from detected positions."""
    regions = {}

    # Default sizes
    NAME_W = 175
    NAME_H = 18
    STACK_DY = 25

    # Player regions
    seat_order = ['hero', 'br', 'tr', 'tc', 'tl', 'bl']
    position_names = ["Hero (du)", "Hoger", "Hoger-upp", "Uppe", "Vanster-upp", "Vanster"]

    player_regions = []
    for idx, seat in enumerate(seat_order):
        p = players[seat]

        if p['name']:
            nx = p['name']['x'] - 5
            ny = p['name']['y']
            nw = max(NAME_W, p['name']['w'] + 10)
            nh = max(NAME_H, p['name']['h'])
        elif p['stack']:
            # If no name found, put name box above stack
            nx = p['stack']['x'] - 5
            ny = p['stack']['y'] - STACK_DY
            nw = max(NAME_W, p['stack']['w'] + 10)
            nh = NAME_H
        else:
            # Use default position (fallback)
            nx, ny, nw, nh = 0, 0, 0, 0

        if p['stack']:
            sx = p['stack']['x'] - 5
            sy = p['stack']['y']
            sw = max(NAME_W, p['stack']['w'] + 10)
            sh = max(NAME_H, p['stack']['h'])
        elif p['name']:
            sx = p['name']['x'] - 5
            sy = p['name']['y'] + STACK_DY
            sw = nw
            sh = NAME_H
        else:
            sx, sy, sw, sh = 0, 0, 0, 0

        player_regions.append({
            "_position": position_names[idx],
            "name": [nx, ny, nw, nh],
            "stack": [sx, sy, sw, sh],
            "bet": [0, 0, 0, 0],
        })

    regions["player_regions"] = player_regions

    # Pot text
    if pot_info:
        regions["pot_text"] = [pot_info['x'] - 5, pot_info['y'] - 2,
                               max(200, pot_info['w'] + 60), pot_info['h'] + 4]
    else:
        # Default pot position (center of window)
        px = win_x + int(0.44 * win_w)
        py = win_y + int(0.32 * win_h)
        regions["pot_text"] = [px, py, 150, 25]

    # Hero cards (estimated relative to hero name position)
    hero = players['hero']
    if hero['name']:
        hx = hero['name']['x']
        hy = hero['name']['y'] - 90  # Cards are above name
        regions["hero_card_1"] = [hx - 10, hy, 50, 70]
        regions["hero_card_2"] = [hx + 50, hy, 50, 70]
    else:
        cx = win_x + int(0.42 * win_w)
        cy = win_y + int(0.55 * win_h)
        regions["hero_card_1"] = [cx, cy, 50, 70]
        regions["hero_card_2"] = [cx + 60, cy, 50, 70]

    # Community cards (center of table, between pot and hero)
    pot_y = regions["pot_text"][1] + regions["pot_text"][3]
    hero_card_y = regions["hero_card_1"][1]
    cc_y = pot_y + 5
    cc_h = 70
    cc_w = 48
    cc_start_x = win_x + int(0.345 * win_w)
    cc_gap = int(0.060 * win_w)
    community = []
    for i in range(5):
        community.append([cc_start_x + i * cc_gap, cc_y, cc_w, cc_h])
    regions["community_cards"] = community

    # Action buttons
    regions["action_buttons_area"] = [
        win_x + int(0.66 * win_w),
        win_y + int(0.87 * win_h),
        int(0.30 * win_w),
        int(0.07 * win_h),
    ]
    regions["dealer_button_search_area"] = [win_x, win_y, win_w, win_h]

    # Metadata
    regions["_poker_window"] = [win_x, win_y, win_w, win_h]
    regions["_screen_capture_region"] = {
        "top": win_y, "left": win_x, "width": win_w, "height": win_h,
    }

    return regions


def main():
    print("=" * 55)
    print("  POKERSTARS AUTO-KALIBRERING v2")
    print("=" * 55)
    print()
    print("Hittar PokerStars automatiskt och laser alla positioner.")
    print("Inga klick behvos!")
    print()

    # Find window
    ps = find_pokerstars_window()
    if not ps:
        print("POKERSTARS HITTADES INTE!")
        print("Oppna ett pokerbord forst.")
        return

    title, win_x, win_y, win_w, win_h = ps
    win_x = max(0, win_x)
    win_y = max(0, win_y)
    print(f"  Fonster: {win_w}x{win_h} vid ({win_x},{win_y})")

    # Capture screen
    print("\nFangar skarmen...")
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Find all player positions
    print("Soker efter spelare med OCR (kan ta 10-15 sek)...")
    players, pot_info = find_player_regions(frame, win_x, win_y, win_w, win_h)

    # Report findings
    print("\nHittade:")
    for seat, p in players.items():
        name = p['name']['text'] if p['name'] else '?'
        stack = p['stack']['text'] if p['stack'] else '?'
        print(f"  {seat:5s}: {name:20s} stack={stack}")

    if pot_info:
        print(f"  pot:   {pot_info['text']}")

    # Build regions
    regions = build_regions(players, pot_info, win_x, win_y, win_w, win_h)

    # Save
    output_path = os.path.join(_dir, "models", "table_regions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(regions, f, indent=2)

    print(f"\nSparad: {output_path}")
    print(f"  Spelare hittade: {sum(1 for p in players.values() if p['name'] or p['stack'])}/6")

    # Verification: try reading each region back
    print("\n=== VERIFIERING ===")
    x, y, w, h = regions['pot_text']
    crop = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7').strip()
    print(f"Pot: \"{text}\"")

    for i, p in enumerate(regions['player_regions']):
        nx, ny, nw, nh = p['name']
        sx, sy, sw, sh = p['stack']
        if nw == 0:
            print(f"P{i} ({p['_position']}): EJ HITTAT")
            continue

        crop_n = frame[max(0,ny):max(1,ny+nh), max(0,nx):max(1,nx+nw)]
        if crop_n.size > 0:
            gray_n = cv2.cvtColor(crop_n, cv2.COLOR_BGR2GRAY)
            _, thr_n = cv2.threshold(gray_n, 150, 255, cv2.THRESH_BINARY)
            name_t = pytesseract.image_to_string(thr_n, config='--psm 7').strip()
        else:
            name_t = ""

        crop_s = frame[max(0,sy):max(1,sy+sh), max(0,sx):max(1,sx+sw)]
        if crop_s.size > 0:
            gray_s = cv2.cvtColor(crop_s, cv2.COLOR_BGR2GRAY)
            _, thr_s = cv2.threshold(gray_s, 150, 255, cv2.THRESH_BINARY)
            stack_t = pytesseract.image_to_string(thr_s, config='--psm 7').strip()
        else:
            stack_t = ""

        print(f"P{i} ({p['_position']}): name=\"{name_t}\", stack=\"{stack_t}\"")

    # Save verification overlay
    eff_w = min(win_w, 1920 - win_x)
    eff_h = min(win_h, 1080 - win_y)
    overlay = frame[win_y:win_y+eff_h, win_x:win_x+eff_w].copy()

    for i, p in enumerate(regions['player_regions']):
        nx, ny, nw, nh = p['name']
        sx, sy, sw, sh = p['stack']
        if nw > 0:
            cv2.rectangle(overlay, (nx-win_x,ny-win_y), (nx-win_x+nw,ny-win_y+nh), (0,0,255), 2)
            cv2.rectangle(overlay, (sx-win_x,sy-win_y), (sx-win_x+sw,sy-win_y+sh), (255,0,0), 2)
            cv2.putText(overlay, f"P{i}", (nx-win_x,ny-win_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    x, y, w, h = regions['pot_text']
    cv2.rectangle(overlay, (x-win_x,y-win_y), (x-win_x+w,y-win_y+h), (0,255,255), 2)

    for cc in regions['community_cards']:
        x, y, w, h = cc
        cv2.rectangle(overlay, (x-win_x,y-win_y), (x-win_x+w,y-win_y+h), (0,200,255), 2)

    for key in ['hero_card_1', 'hero_card_2']:
        x, y, w, h = regions[key]
        cv2.rectangle(overlay, (x-win_x,y-win_y), (x-win_x+w,y-win_y+h), (0,255,0), 2)

    out = os.path.join(_dir, 'debug_images', 'auto_cal_overlay.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    success, buf = cv2.imencode('.png', overlay)
    if success:
        with open(out, 'wb') as f:
            f.write(buf.tobytes())
    print(f"\nOverlay: {out}")
    print("\nKLAR! Kor 'python main.py' for att testa.")


if __name__ == "__main__":
    main()
