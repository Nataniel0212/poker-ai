"""
Capture card templates from a live PokerStars table.

Uses calibrated regions to auto-crop cards from hero hand and community cards.
Shows each card and lets you label it (e.g., "Ah" for Ace of hearts).

Usage:
    python tools/capture_card_templates.py

Controls:
    - Type card name (e.g., Ah, Ks, Td, 2c) and press Enter to save
    - Press Enter without typing to skip a card
    - Type 'q' to quit
    - Type 'n' to capture next hand (new screenshot)
"""

import sys
import os
import io
import time
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mss
import ctypes
import ctypes.wintypes

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']
ALL_CARDS = {f"{r}{s}" for r in RANKS for s in SUITS}

RANK_NAMES = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
    '8': '8', '9': '9', 'T': '10/T', 'J': 'J/Knekt', 'Q': 'Q/Dam',
    'K': 'K/Kung', 'A': 'A/Ess'
}
SUIT_NAMES = {'s': 'Spader', 'h': 'Hjartan', 'd': 'Ruter', 'c': 'Klover'}


def find_pokerstars_window(bring_to_front=True):
    """Find PokerStars window using Windows API."""
    user32 = ctypes.windll.user32
    results = []

    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            title = buf.value
            keywords = ['poker', 'holdem', "hold'em", 'gadolin', 'tournament']
            if any(k in title.lower() for k in keywords):
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                x, y = rect.left, rect.top
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 400 and h > 300:
                    if bring_to_front:
                        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                        user32.SetForegroundWindow(hwnd)
                    results.append((title, x, y, w, h))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int))
    user32.EnumWindows(WNDENUMPROC(callback), 0)

    if results:
        # Pick largest window
        results.sort(key=lambda r: r[3] * r[4], reverse=True)
        return results[0]
    return None


def capture_screen():
    """Capture full screen."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def is_card_visible(roi):
    """Check if a card is visible in the ROI (not empty/green felt)."""
    if roi is None or roi.size == 0:
        return False

    # Cards are mostly white/light with colored symbols
    # Green felt is... green
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Check if mostly white/light (card background)
    white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
    white_pct = np.sum(white_mask > 0) / white_mask.size

    # Also check for non-green colors (card has red/black/blue elements)
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    green_pct = np.sum(green_mask > 0) / green_mask.size

    # Card visible if significant white AND not mostly green
    return white_pct > 0.3 and green_pct < 0.5


def show_card_large(roi, label="Card"):
    """Display a card ROI enlarged for easy identification."""
    # Scale up 4x for visibility
    large = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(label, large)
    cv2.waitKey(1)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    regions_file = os.path.join(base_dir, "models", "table_regions.json")
    output_dir = os.path.join(base_dir, "models", "card_templates")

    os.makedirs(output_dir, exist_ok=True)

    # Load regions
    if not os.path.exists(regions_file):
        print("FEL: Inga kalibrerade regioner hittades!")
        print("Kor forst: python calibrate_pokerstars.py")
        return

    with open(regions_file) as f:
        regions = json.load(f)

    # Check which cards we already have
    existing = set()
    for r in RANKS:
        for s in SUITS:
            card = f"{r}{s}"
            if os.path.exists(os.path.join(output_dir, f"{card}.png")):
                existing.add(card)

    missing = ALL_CARDS - existing

    print("=" * 55)
    print("  KORT-TEMPLATE FANGARE")
    print("=" * 55)
    print()
    print(f"  Redan fangade: {len(existing)}/52")
    print(f"  Saknas:        {len(missing)}/52")
    print()
    print("  Sa har fungerar det:")
    print("  1. Spela poker pa PokerStars (play money)")
    print("  2. Skriptet fangar dina kort + community cards")
    print("  3. Skriv kortnamnet (t.ex. Ah, Ks, Td, 2c)")
    print("  4. Tryck Enter for att spara, tom rad for att skippa")
    print("  5. Skriv 'n' for att ta ny skarmfangst (nasta hand)")
    print("  6. Skriv 'q' for att avsluta")
    print()
    print("  Kortformat: Rank + Suit")
    print("    Rank: 2-9, T(10), J, Q, K, A")
    print("    Suit: s(spader), h(hjartan), d(ruter), c(klover)")
    print()

    # All card positions (hero + community)
    card_regions = [
        ("Hero kort 1", regions["hero_card_1"]),
        ("Hero kort 2", regions["hero_card_2"]),
    ]
    for i, cc in enumerate(regions["community_cards"]):
        card_regions.append((f"Community {i+1}", cc))

    saved_count = 0

    while len(missing) > 0:
        print(f"\n--- Fangar skarm (saknas: {len(missing)} kort) ---")
        print("Se till att PokerStars visar kort...")

        # Bring PS to front and capture
        ps = find_pokerstars_window(bring_to_front=True)
        if ps is None:
            print("Kunde inte hitta PokerStars-fonstret!")
            input("Oppna PokerStars och tryck Enter...")
            continue

        time.sleep(0.3)
        frame = capture_screen()

        # Check each card position
        visible_cards = []
        for label, region in card_regions:
            x, y, w, h = region
            roi = frame[y:y+h, x:x+w]
            if is_card_visible(roi):
                visible_cards.append((label, roi.copy(), region))

        if not visible_cards:
            print("Inga synliga kort hittades. Vanta tills en hand delas ut.")
            cmd = input("Tryck Enter for att forsoka igen, 'q' for att avsluta: ").strip()
            if cmd.lower() == 'q':
                break
            continue

        print(f"Hittade {len(visible_cards)} synliga kort!\n")

        for label, roi, region in visible_cards:
            # Show card enlarged
            show_card_large(roi, label)

            print(f"  {label} (position {region[0]},{region[1]}):")
            print(f"    Saknade kort: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            cmd = input(f"    Skriv kortnamn (t.ex. Ah), Enter=skippa, n=nasta hand, q=avsluta: ").strip()

            cv2.destroyWindow(label)

            if cmd.lower() == 'q':
                missing = set()  # Break outer loop too
                break
            if cmd.lower() == 'n':
                break
            if not cmd:
                continue

            # Normalize input
            card_name = cmd.strip()
            # Handle common inputs: "10h" -> "Th", "ah" -> "Ah"
            if card_name.startswith("10"):
                card_name = "T" + card_name[2:]
            if len(card_name) == 2:
                card_name = card_name[0].upper() + card_name[1].lower()

            if card_name not in ALL_CARDS:
                print(f"    Ogiltigt kort: '{card_name}'. Format: Rank(2-9,T,J,Q,K,A) + Suit(s,h,d,c)")
                continue

            if card_name in existing:
                overwrite = input(f"    {card_name} finns redan. Skriva over? (j/n): ").strip()
                if overwrite.lower() != 'j':
                    continue

            # Save template
            filepath = os.path.join(output_dir, f"{card_name}.png")
            cv2.imwrite(filepath, roi)
            existing.add(card_name)
            missing.discard(card_name)
            saved_count += 1
            rank_name = RANK_NAMES.get(card_name[0], card_name[0])
            suit_name = SUIT_NAMES.get(card_name[1], card_name[1])
            print(f"    Sparat: {card_name} ({rank_name} {suit_name}) -> {filepath}")

        # Small pause before next capture
        if missing:
            cmd = input("\nTryck Enter for nasta fangst, 'q' for att avsluta: ").strip()
            if cmd.lower() == 'q':
                break

    cv2.destroyAllWindows()

    print(f"\n{'=' * 55}")
    print(f"  KLART!")
    print(f"  Sparade {saved_count} nya templates denna session")
    print(f"  Totalt: {len(existing)}/52 kort")
    if missing:
        print(f"  Saknas fortfarande: {sorted(missing)}")
    else:
        print(f"  Alla 52 kort fangade!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
