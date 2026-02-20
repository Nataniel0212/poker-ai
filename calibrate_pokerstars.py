"""
Auto-calibration for PokerStars 6-max tables.

NO CLICKS NEEDED — automatically detects the green poker table
and calculates all element positions.

Just run: python calibrate_pokerstars.py
"""

import sys
import os
import io
import json
import time

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

# Debug images directory
DEBUG_DIR = os.path.join(_dir, "debug_images")
os.makedirs(DEBUG_DIR, exist_ok=True)


def save_image(path, img):
    """Save image with Unicode path support (cv2.imwrite fails on å/ä/ö)."""
    success, buf = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())


def auto_detect_layout(frame):
    """Auto-detect card positions and table layout from a live frame.

    Finds white card rectangles on the green felt, clusters them into
    community cards and hero cards, then derives all other positions
    (pot, players, buttons) relative to the detected cards.

    Returns dict with all regions, or None if detection fails.
    """
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find white/bright rectangles (card candidates)
    _, white_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_rects = []
    all_rects = []  # For debug
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)
        area = cw * ch
        # Card should be roughly 35-150px wide, 50-200px tall
        if 35 < cw < 150 and 50 < ch < 200 and 0.40 < aspect < 0.95 and area > 1500:
            card_rects.append((x, y, cw, ch))
            all_rects.append((x, y, cw, ch, aspect, area, "PASS"))
        elif area > 1000 and 20 < cw < 200 and 30 < ch < 250:
            all_rects.append((x, y, cw, ch, aspect, area, f"FAIL(a={aspect:.2f})"))
    card_rects.sort(key=lambda r: r[0])

    # Debug: show all candidate rects
    if len(card_rects) < 3:
        for r in sorted(all_rects, key=lambda r: r[5], reverse=True)[:10]:
            x, y, cw, ch, asp, area, status = r
            print(f"    rect ({x},{y},{cw},{ch}) aspect={asp:.2f} area={area} {status}")

    if len(card_rects) < 2:
        print(f"  Auto-detect: only {len(card_rects)} card rectangles found, need at least 2")
        return None

    # Cluster cards by y-position (community vs hero)
    y_groups = {}
    for rect in card_rects:
        x, y, cw, ch = rect
        placed = False
        for gy in list(y_groups.keys()):
            if abs(y - gy) < ch * 0.5:  # Same row if within half a card height
                y_groups[gy].append(rect)
                placed = True
                break
        if not placed:
            y_groups[y] = [rect]

    # Find community cards: group with 3-5 cards in a row
    comm_cards = None
    hero_cards = None
    for gy in sorted(y_groups.keys()):
        group = y_groups[gy]
        if 3 <= len(group) <= 5 and comm_cards is None:
            comm_cards = sorted(group, key=lambda r: r[0])
        elif len(group) == 2 and comm_cards is not None:
            # Hero cards: 2 cards below community cards
            hero_cards = sorted(group, key=lambda r: r[0])

    if comm_cards is None:
        print(f"  Auto-detect: no community card row found (groups: {[len(g) for g in y_groups.values()]})")
        return None

    # === Extract measurements from community cards ===
    comm_y = comm_cards[0][1]
    comm_first_x = comm_cards[0][0]
    card_w = int(np.mean([c[2] for c in comm_cards]))
    card_h = int(np.mean([c[3] for c in comm_cards]))

    if len(comm_cards) > 1:
        gaps = [comm_cards[i+1][0] - comm_cards[i][0] for i in range(len(comm_cards)-1)]
        card_gap = int(np.mean(gaps))
    else:
        card_gap = int(card_w * 1.15)

    # Use detected card positions directly instead of extrapolating from center.
    # Detected cards might be the flop (first 3), not the middle of 5.
    # Place detected cards at their actual positions, extrapolate the rest.
    comm_start_x = comm_cards[0][0]  # First detected card IS the first visible card

    # Table center: estimate from first card + 2 gaps (center of 5 cards)
    table_center_x = comm_start_x + 2 * card_gap + card_w // 2

    print(f"  Community: {len(comm_cards)} cards at y={comm_y}, center_x={table_center_x}")
    print(f"  Card size: {card_w}x{card_h}, gap={card_gap}")

    # === Hero cards ===
    if hero_cards:
        hero_y = hero_cards[0][1]
        hero_x1 = hero_cards[0][0]
        hero_x2 = hero_cards[1][0]
        hero_w = int(np.mean([c[2] for c in hero_cards]))
        hero_h = int(np.mean([c[3] for c in hero_cards]))
        print(f"  Hero cards: 2 cards at y={hero_y}, x=[{hero_x1},{hero_x2}]")
    else:
        # Estimate hero card position: ~2.0 card heights below community
        # PS hero cards are further down (below the pot text and community area)
        hero_y = comm_y + int(card_h * 2.0)
        hero_w = card_w
        hero_h = card_h
        hero_x1 = table_center_x - card_gap // 2 - card_w
        hero_x2 = table_center_x + card_gap // 2
        print(f"  Hero cards: estimated at y={hero_y}")

    # === Pot text: just above community cards ===
    # PS shows "Pott: X XXX" centered above the cards, ~0.7 card heights up
    pot_y = comm_y - int(card_h * 0.8)
    pot_x = table_center_x - int(card_gap * 2)
    pot_w = int(card_gap * 4)
    pot_h = int(card_h * 0.45)

    # === Build regions dict ===
    border = 5  # Small border around card ROIs for template matching slack
    regions = {}
    regions["hero_card_1"] = [hero_x1 - border, hero_y - border,
                               hero_w + 2*border, hero_h + 2*border]
    regions["hero_card_2"] = [hero_x2 - border, hero_y - border,
                               hero_w + 2*border, hero_h + 2*border]

    community = []
    for i in range(5):
        if i < len(comm_cards):
            # Use actual detected position for known cards
            cx = comm_cards[i][0]
        else:
            # Extrapolate remaining positions from last detected card
            last_x = comm_cards[-1][0]
            cx = last_x + (i - len(comm_cards) + 1) * card_gap
        community.append([cx - border, comm_y - border,
                         card_w + 2*border, card_h + 2*border])
    regions["community_cards"] = community

    regions["pot_text"] = [pot_x, pot_y, pot_w, pot_h]

    # === Player positions (6-max, relative to table center and card positions) ===
    # Estimated from standard PS 6-max layout
    table_left = comm_start_x - int(card_gap * 2)
    table_right = comm_start_x + 5 * card_gap + int(card_gap * 2)
    table_top = comm_y - int(card_h * 3)
    table_bottom = hero_y + hero_h + int(card_h * 2)
    name_w = int((table_right - table_left) * 0.18)
    name_h = int(card_h * 0.35)

    player_regions = []
    seat_positions = [
        # Seat 0: Hero (bottom center)
        (table_center_x - name_w//2, hero_y + hero_h + int(card_h * 0.5)),
        # Seat 1: Bottom right
        (table_right - name_w, hero_y - int(card_h * 0.5)),
        # Seat 2: Top right
        (table_right - name_w, table_top + int(card_h * 0.5)),
        # Seat 3: Top center
        (table_center_x - name_w//2, table_top - name_h),
        # Seat 4: Top left
        (table_left, table_top + int(card_h * 0.5)),
        # Seat 5: Bottom left
        (table_left, hero_y - int(card_h * 0.5)),
    ]
    names = ["Hero", "Hoger", "Hoger-upp", "Uppe", "Vanster-upp", "Vanster"]
    for i, (px, py) in enumerate(seat_positions):
        # Clamp to frame bounds
        px = max(0, min(px, fw - name_w))
        py = max(0, min(py, fh - name_h))

        stack_y = py + name_h + 2
        bet_x = int((px + table_center_x) / 2)  # Between player and center
        bet_y = int((py + comm_y) / 2)

        player_regions.append({
            "_position": names[i],
            "name": [px, py, name_w, name_h],
            "stack": [px, stack_y, name_w, name_h],
            "bet": [bet_x, bet_y, int(name_w * 0.6), name_h],
        })
    regions["player_regions"] = player_regions

    # Action buttons (bottom right)
    regions["action_buttons_area"] = [int(fw * 0.65), int(fh * 0.87), int(fw * 0.32), int(fh * 0.07)]

    # Dealer button search areas
    dealer_regions = []
    for i, (px, py) in enumerate(seat_positions):
        dx = int((px + table_center_x) / 2)
        dy = int((py + comm_y) / 2)
        dealer_regions.append([dx, dy, int(card_w * 1.5), int(card_h * 0.6)])
    regions["dealer_button_regions"] = dealer_regions
    regions["dealer_button_search_area"] = [table_left, table_top,
                                             table_right - table_left,
                                             table_bottom - table_top]

    # Store detection metadata
    regions["_auto_detected"] = True
    regions["_card_size"] = [card_w, card_h]
    regions["_card_gap"] = card_gap
    regions["_table_center_x"] = table_center_x
    regions["_comm_y"] = comm_y
    regions["_hero_y"] = hero_y

    return regions


def find_pokerstars_window(bring_to_front=False):
    """Find PokerStars window using Windows API.
    If bring_to_front=True, brings the window to the foreground."""
    import ctypes
    from ctypes import wintypes

    # Enable DPI awareness so we get physical pixel coordinates
    # (mss captures physical pixels, so coordinates must match)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

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
                        windows.append((title, x, y, w, h, hwnd))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_handler), 0)

    # Look for PokerStars table window — prioritize actual game tables
    # Use GetClientRect for accurate content area (excludes title bar/borders)
    def get_client_rect(hwnd):
        """Get client area position and size (excludes title bar and borders)."""
        client_rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(client_rect))
        # Convert client (0,0) to screen coords
        point = wintypes.POINT(0, 0)
        ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(point))
        return (point.x, point.y, client_rect.right, client_rect.bottom)

    # Priority 1: Hold'em/Omaha table (has "hold" or "omaha" AND "limit")
    # Must be landscape (wider than tall) to be an actual game table
    found = None
    for title, x, y, w, h, hwnd in windows:
        tl = title.lower()
        if ('hold' in tl or 'omaha' in tl) and ('limit' in tl or 'no limit' in tl):
            cx, cy, cw, ch = get_client_rect(hwnd)
            if cw > ch * 0.9:  # Table should be roughly landscape
                found = (title, cx, cy, cw, ch, hwnd)
                break

    # Priority 2: Window with poker-related name (landscape only)
    if not found:
        for title, x, y, w, h, hwnd in windows:
            tl = title.lower()
            if any(kw in tl for kw in ['gadolin', 'kalevala', 'poker table', 'stars']) and 'calibr' not in tl:
                cx, cy, cw, ch = get_client_rect(hwnd)
                if cw > ch * 0.9:
                    found = (title, cx, cy, cw, ch, hwnd)
                    break

    # Priority 3: Any PokerStars window (lobby etc)
    if not found:
        for title, x, y, w, h, hwnd in windows:
            tl = title.lower()
            if 'pokerstars' in tl and w > 400 and 'calibr' not in tl:
                cx, cy, cw, ch = get_client_rect(hwnd)
                found = (title, cx, cy, cw, ch, hwnd)
                break

    if found and bring_to_front:
        hwnd = found[5]
        # Bring window to foreground
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        # Wait for window to settle (Windows Snap may resize/move it)
        import time
        time.sleep(1.0)
        # Re-read coordinates AFTER the window has settled
        cx, cy, cw, ch = get_client_rect(hwnd)
        title = found[0]
        found = (title, cx, cy, cw, ch, hwnd)

    if found:
        return found[:5]  # Return without hwnd
    return None


def find_green_table(frame):
    """Auto-detect the green poker table using color detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 200])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10000:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


def pct_to_pixels(pct_region, table_x, table_y, table_w, table_h):
    """Convert percentage-based region to pixel coordinates."""
    px, py, pw, ph = pct_region
    return [
        int(table_x + px * table_w),
        int(table_y + py * table_h),
        int(pw * table_w),
        int(ph * table_h),
    ]


def calculate_regions(table_x, table_y, table_w, table_h):
    """Fallback: calculate regions from window position using rough proportions.

    Only used when auto_detect_layout() fails (no cards visible).
    These proportions are approximate and may need adjustment per window.
    """
    regions = {}

    # Very rough fallback — prefer auto_detect_layout() instead
    regions["hero_card_1"] = [int(table_x + table_w * 0.42), int(table_y + table_h * 0.78),
                               int(table_w * 0.06), int(table_h * 0.11)]
    regions["hero_card_2"] = [int(table_x + table_w * 0.49), int(table_y + table_h * 0.78),
                               int(table_w * 0.06), int(table_h * 0.11)]

    community = []
    for i in range(5):
        cx = 0.30 + i * 0.065
        community.append([int(table_x + cx * table_w), int(table_y + table_h * 0.58),
                          int(table_w * 0.065), int(table_h * 0.13)])
    regions["community_cards"] = community
    regions["pot_text"] = [int(table_x + table_w * 0.35), int(table_y + table_h * 0.50),
                           int(table_w * 0.25), int(table_h * 0.05)]

    names = ["Hero", "Hoger", "Hoger-upp", "Uppe", "Vanster-upp", "Vanster"]
    player_regions = []
    for i, name in enumerate(names):
        player_regions.append({
            "_position": name,
            "name": [0, 0, 1, 1],
            "stack": [0, 0, 1, 1],
            "bet": [0, 0, 1, 1],
        })
    regions["player_regions"] = player_regions
    regions["action_buttons_area"] = [0, 0, 1, 1]
    regions["dealer_button_regions"] = [[0, 0, 1, 1]] * 6
    regions["dealer_button_search_area"] = [table_x, table_y, table_w, table_h]

    return regions


def draw_regions(frame, regions, table_rect):
    """Draw all detected regions on frame for verification."""
    display = frame.copy()

    # Table area
    tx, ty, tw, th = table_rect
    cv2.rectangle(display, (tx, ty), (tx+tw, ty+th), (255, 255, 0), 2)
    cv2.putText(display, "TABLE", (tx+5, ty+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Hero cards
    for label, key in [("CARD1", "hero_card_1"), ("CARD2", "hero_card_2")]:
        x, y, w, h = regions[key]
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Community cards
    for i, (x, y, w, h) in enumerate(regions["community_cards"]):
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 200, 255), 2)
        cv2.putText(display, f"CC{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # Pot text
    x, y, w, h = regions["pot_text"]
    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(display, "POT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Players
    for i, p in enumerate(regions["player_regions"]):
        nx, ny, nw, nh = p["name"]
        sx, sy, sw, sh = p["stack"]
        bx, by, bw, bh = p["bet"]
        cv2.rectangle(display, (nx, ny), (nx+nw, ny+nh), (255, 100, 100), 2)
        cv2.rectangle(display, (sx, sy), (sx+sw, sy+sh), (100, 100, 255), 2)
        if bw > 0 and bh > 0:
            cv2.rectangle(display, (bx, by), (bx+bw, by+bh), (200, 200, 0), 1)
        cv2.putText(display, f"P{i}", (nx, ny-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

    # Dealer button search regions
    if "dealer_button_regions" in regions:
        for i, (dx, dy, dw, dh) in enumerate(regions["dealer_button_regions"]):
            cv2.rectangle(display, (dx, dy), (dx+dw, dy+dh), (255, 255, 255), 1)
            cv2.putText(display, f"D{i}", (dx, dy-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return display


def main():
    print("=" * 55)
    print("  POKERSTARS AUTO-KALIBRERING")
    print("=" * 55)
    print()
    print("Detekterar kort-positioner automatiskt fran live-bordet.")
    print("OBS: Kort maste vara synliga pa bordet (community cards)!")
    print()

    # Find PokerStars window first and bring to front
    print("Letar efter PokerStars-fonster...")
    ps_window = find_pokerstars_window(bring_to_front=True)

    if ps_window is None:
        print("KUNDE INTE HITTA POKERSTARS!")
        print("Se till att ett pokerbord ar oppet i PokerStars.")
        return

    win_title, win_x, win_y, win_w, win_h = ps_window
    win_x = max(0, win_x)
    win_y = max(0, win_y)
    print(f"  Fonster: \"{win_title}\"")
    print(f"  Storlek: {win_w}x{win_h} vid ({win_x},{win_y})")

    print("  Bringar PokerStars till forgrunden...")
    time.sleep(1.5)  # Wait for window to come to front

    # Capture just the PS window
    print("Fangar fonster...")
    with mss.mss() as sct:
        region = {"top": win_y, "left": win_x, "width": win_w, "height": win_h}
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    # Try auto-detection first
    print("\nAuto-detekterar kort-positioner...")
    regions = auto_detect_layout(frame)

    if regions is None:
        print("\nINGA KORT HITTADES! Forsoker med gron-bords-detektion...")
        regions = calculate_regions(0, 0, win_w, win_h)
        print("  Anvander grova fallback-proportioner.")
        print("  Kor om nar kort ar synliga pa bordet!")

    # Save
    regions["_poker_window"] = [win_x, win_y, win_w, win_h]
    regions["_screen_capture_region"] = {
        "top": win_y, "left": win_x, "width": win_w, "height": win_h,
    }

    output_path = os.path.join(_dir, "models", "table_regions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(regions, f, indent=2)

    print("\n" + "=" * 55)
    print("  KALIBRERING KLAR!")
    print("=" * 55)
    print(f"\nSparad till: {output_path}")
    print(f"  Hero kort:       {regions['hero_card_1']} / {regions['hero_card_2']}")
    print(f"  Community cards:  {len(regions['community_cards'])} st")
    print(f"  Pot text:        {regions['pot_text']}")
    print(f"  Spelare:         {len(regions['player_regions'])} st")

    # Show verification image
    print("\nVisar forhandsvisning i 8 sekunder...")
    table_rect = [0, 0, win_w, win_h]
    verification = draw_regions(frame, regions, table_rect)

    overlay = verification.copy()
    cv2.rectangle(overlay, (0, 0), (700, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, verification, 0.3, 0, verification)
    status = "AUTO-DETEKTERAD" if regions.get("_auto_detected") else "FALLBACK"
    cv2.putText(verification, f"KALIBRERING {status}! Stammer rutorna? (8s)",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Kalibrering", verification)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()

    os.makedirs(os.path.join(_dir, "debug_images"), exist_ok=True)
    verify_path = os.path.join(_dir, "debug_images", "calibration_preview.png")
    save_image(verify_path, verification)
    print(f"\nForhandsvisning sparad: {verify_path}")


def main_manual():
    """Manual calibration with 2 clicks as fallback."""
    print("=" * 55)
    print("  POKERSTARS MANUELL KALIBRERING")
    print("=" * 55)
    print()
    print("Klicka TVA ganger:")
    print("  1) Uppe-vanster hornet av det GRONA bordet")
    print("  2) Nere-hoger hornet av det GRONA bordet")
    print()
    print("Kalibreringen sparas AUTOMATISKT efter 2 klick.")
    print()

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    points = []
    display = frame.copy()

    # Clear instructions
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (800, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    cv2.putText(display, "KLICK 1: Uppe-vanster hornet av GRONA bordet",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display, "KLICK 2: Nere-hoger hornet av GRONA bordet  |  Q=avbryt",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    saved = [False]

    def click_handler(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or saved[0]:
            return
        points.append((x, y))
        cv2.circle(display, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(display, f"Klick {len(points)}", (x+15, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Kalibrering", display)
        print(f"  Klick {len(points)}: ({x}, {y})")

    cv2.imshow("Kalibrering", display)
    cv2.setMouseCallback("Kalibrering", click_handler)

    start = time.time()
    while len(points) < 2:
        key = cv2.waitKey(100)
        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            print("Avbrutet.")
            return
        if time.time() - start > 120:
            cv2.destroyAllWindows()
            print("Timeout!")
            return

    saved[0] = True  # Prevent more clicks

    (x1, y1), (x2, y2) = points
    table_x = min(x1, x2)
    table_y = min(y1, y2)
    table_w = abs(x2 - x1)
    table_h = abs(y2 - y1)

    print(f"\n  Bordarea: ({table_x}, {table_y}) - storlek {table_w}x{table_h}")

    regions = calculate_regions(table_x, table_y, table_w, table_h)
    regions["_poker_window"] = [table_x, table_y, table_w, table_h]
    regions["_screen_capture_region"] = {
        "top": table_y,
        "left": table_x,
        "width": table_w,
        "height": table_h,
    }

    output_path = os.path.join(_dir, "models", "table_regions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(regions, f, indent=2)

    print(f"\n  SPARAD till: {output_path}")

    # Show verification for 8 seconds
    verification = draw_regions(frame, regions, [table_x, table_y, table_w, table_h])
    overlay = verification.copy()
    cv2.rectangle(overlay, (0, 0), (500, 50), (0, 100, 0), -1)
    cv2.addWeighted(overlay, 0.7, verification, 0.3, 0, verification)
    cv2.putText(verification, "SPARAD! Visar i 8 sek...",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Kalibrering", verification)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()

    os.makedirs(os.path.join(_dir, "debug_images"), exist_ok=True)
    verify_path = os.path.join(_dir, "debug_images", "calibration_preview.png")
    save_image(verify_path, verification)
    print(f"  Forhandsvisning sparad: {verify_path}")


if __name__ == "__main__":
    if "--manual" in sys.argv:
        main_manual()
    else:
        main()
