"""
Card Template Capture Tool — saves card ROIs with temp names.
After capture, run 'python capture_cards.py --verify' to see all captured cards.

Usage:
  python capture_cards.py              # Capture cards while you play
  python capture_cards.py --verify     # Show all captured temps for verification
"""

import sys
import io
import os
import time
import hashlib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
os.chdir(_dir)

import cv2
import numpy as np
import mss

from calibrate_pokerstars import find_pokerstars_window, calculate_regions

TEMPLATE_DIR = os.path.join(_dir, "models", "card_templates_live")
TEMP_DIR = os.path.join(_dir, "models", "card_templates_temp")
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def save_image(path, img):
    success, buf = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())


def image_hash(roi):
    """Create a perceptual hash of a card ROI to detect duplicates."""
    small = cv2.resize(roi, (16, 16))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()[:8]


def is_card_present(roi):
    if roi is None or roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 220]))
    green_ratio = cv2.countNonZero(green) / max(roi.shape[0] * roi.shape[1], 1)
    if green_ratio > 0.50:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 30:
        return False
    # Card should have a significant white/light area (card face)
    white = cv2.inRange(gray, 180, 255)
    white_ratio = cv2.countNonZero(white) / max(roi.shape[0] * roi.shape[1], 1)
    if white_ratio < 0.15:
        return False
    return True


def capture_mode():
    """Capture card ROIs continuously while playing."""
    print("=" * 55)
    print("  CARD TEMPLATE CAPTURE")
    print("=" * 55)
    print()

    # Count existing
    existing = len([f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')])
    temps = len([f for f in os.listdir(TEMP_DIR) if f.endswith('.png')])
    print(f"Verified templates: {existing}/52")
    print(f"Temp captures:      {temps}")
    print()

    ps = find_pokerstars_window()
    if not ps:
        print("PokerStars not found!")
        return

    title, wx, wy, ww, wh = ps
    wx, wy = max(0, wx), max(0, wy)
    print(f"Found: {ww}x{wh} at ({wx},{wy})")

    calc = calculate_regions(0, 0, ww, wh)
    all_regions = [
        ("H1", calc["hero_card_1"]),
        ("H2", calc["hero_card_2"]),
    ]
    for i, cc in enumerate(calc["community_cards"]):
        all_regions.append((f"CC{i+1}", cc))

    print(f"\nCapturing from {len(all_regions)} positions...")
    print("Play poker! Cards are saved automatically.")
    print("Press Ctrl+C to stop.\n")

    seen_hashes = set()
    # Load hashes of existing temps
    for f in os.listdir(TEMP_DIR):
        if f.endswith('.png'):
            seen_hashes.add(f.split('_')[0] if '_' in f else f.replace('.png', ''))

    captured = 0
    frame_count = 0

    try:
        with mss.mss() as sct:
            region = {"top": wy, "left": wx, "width": ww, "height": wh}

            while True:
                screenshot = sct.grab(region)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame_count += 1

                for label, (rx, ry, rw, rh) in all_regions:
                    roi = frame[ry:ry+rh, rx:rx+rw]
                    if not is_card_present(roi):
                        continue

                    h = image_hash(roi)
                    if h in seen_hashes:
                        continue

                    seen_hashes.add(h)
                    captured += 1
                    filename = f"{h}_{label}.png"
                    save_image(os.path.join(TEMP_DIR, filename), roi)
                    print(f"  #{captured} from {label} ({filename})", flush=True)

                if frame_count % 20 == 0:
                    sys.stdout.write(f"\r  Frame {frame_count}, {captured} new cards captured...  ")
                    sys.stdout.flush()

                time.sleep(0.8)

    except KeyboardInterrupt:
        print(f"\n\nDone! Captured {captured} unique card images.")
        print(f"Run: python capture_cards.py --verify")
        print(f"to see all captures and name them correctly.")


def verify_mode():
    """Show all temp captures for manual verification."""
    print("=" * 55)
    print("  VERIFY CAPTURED CARDS")
    print("=" * 55)
    print()

    temps = sorted([f for f in os.listdir(TEMP_DIR) if f.endswith('.png')])
    if not temps:
        print("No temp captures found! Run: python capture_cards.py first.")
        return

    print(f"Found {len(temps)} temp captures in {TEMP_DIR}")
    print()

    # Create a grid image showing all captures
    images = []
    labels = []
    for f in temps:
        path = os.path.join(TEMP_DIR, f)
        with open(path, 'rb') as fh:
            data = np.frombuffer(fh.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            labels.append(f.replace('.png', ''))

    if not images:
        print("No valid images!")
        return

    # Resize all to same size
    target_h, target_w = 120, 90
    cols = min(8, len(images))
    rows = (len(images) + cols - 1) // cols

    grid = np.ones((rows * (target_h + 25), cols * (target_w + 10), 3), dtype=np.uint8) * 40

    for idx, (img, label) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(img, (target_w, target_h))
        y_off = r * (target_h + 25) + 20
        x_off = c * (target_w + 10) + 5
        grid[y_off:y_off+target_h, x_off:x_off+target_w] = resized
        # Label
        cv2.putText(grid, f"#{idx+1}", (x_off, y_off - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    grid_path = os.path.join(_dir, "debug_images", "captured_cards_grid.png")
    save_image(grid_path, grid)
    print(f"Grid saved to: {grid_path}")
    print()

    # Interactive naming
    print("Name each card (e.g., 'Ah' for Ace of hearts, 'skip' to skip, 'quit' to stop):")
    print("  Ranks: 2 3 4 5 6 7 8 9 T J Q K A")
    print("  Suits: h(hearts) d(diamonds) c(clubs) s(spades)")
    print()

    valid_ranks = set('23456789TJQKA')
    valid_suits = set('hdcs')

    for idx, (f, label) in enumerate(zip(temps, labels)):
        print(f"  Card #{idx+1} (file: {f})")
        while True:
            name = input(f"    Name (e.g. Ah): ").strip()
            if name.lower() == 'quit':
                print("Stopped.")
                return
            if name.lower() == 'skip':
                break
            if len(name) == 2 and name[0] in valid_ranks and name[1] in valid_suits:
                src = os.path.join(TEMP_DIR, f)
                dst = os.path.join(TEMPLATE_DIR, f"{name}.png")
                if os.path.exists(dst):
                    print(f"    {name}.png already exists! Overwrite? (y/n): ", end="")
                    if input().strip().lower() != 'y':
                        continue
                import shutil
                shutil.copy2(src, dst)
                os.remove(src)
                print(f"    Saved as {name}.png")
                break
            else:
                print(f"    Invalid! Use format like 'Ah', 'Ts', '9c'")

    remaining = len([f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')])
    print(f"\nTotal verified templates: {remaining}/52")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_mode()
    else:
        capture_mode()
