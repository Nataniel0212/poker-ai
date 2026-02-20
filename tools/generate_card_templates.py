"""
Card Template Generator — creates synthetic card images for template matching.

Generates all 52 playing cards as template images that can be used
for OpenCV matchTemplate() card detection.

These templates work well for most online poker clients that use
standard card designs. For best results, also capture templates
directly from your poker client using the capture_templates tool.

Usage:
    python tools/generate_card_templates.py
    python tools/generate_card_templates.py --size 50x70
    python tools/generate_card_templates.py --style pokerstars
"""

import os
import sys
import argparse
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
RANK_DISPLAY = {
    '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
    '9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A',
}
SUITS = {
    's': {'name': 'spades',   'color': (0, 0, 0),       'symbol': 'S'},
    'h': {'name': 'hearts',   'color': (200, 0, 0),     'symbol': 'H'},
    'd': {'name': 'diamonds', 'color': (0, 0, 200),     'symbol': 'D'},
    'c': {'name': 'clubs',    'color': (0, 100, 0),     'symbol': 'C'},
}

# PokerStars-style colors
POKERSTARS_SUITS = {
    's': {'name': 'spades',   'color': (30, 30, 30),    'symbol': 'S'},
    'h': {'name': 'hearts',   'color': (180, 20, 20),   'symbol': 'H'},
    'd': {'name': 'diamonds', 'color': (20, 80, 180),   'symbol': 'D'},
    'c': {'name': 'clubs',    'color': (20, 120, 20),   'symbol': 'C'},
}


def generate_card_image(rank: str, suit: str, width: int = 50, height: int = 70,
                         style: str = "default") -> np.ndarray:
    """Generate a single card template image.

    Args:
        rank: Card rank (2-9, T, J, Q, K, A)
        suit: Card suit (s, h, d, c)
        width: Template width in pixels
        height: Template height in pixels
        style: Visual style ("default" or "pokerstars")

    Returns:
        BGR numpy array of the card image
    """
    suits = POKERSTARS_SUITS if style == "pokerstars" else SUITS

    # Create PIL image (white background with rounded corners)
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    suit_info = suits[suit]
    color = suit_info['color']
    rank_text = RANK_DISPLAY[rank]
    suit_text = suit_info['symbol']

    # Try to use a good font, fallback to default
    font_size = max(12, width // 3)
    suit_font_size = max(10, width // 4)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        suit_font = ImageFont.truetype("arial.ttf", suit_font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            suit_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", suit_font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
            suit_font = ImageFont.load_default()

    # Draw rank in top-left
    rank_x = 3
    rank_y = 2
    draw.text((rank_x, rank_y), rank_text, fill=color, font=font)

    # Draw suit symbol below rank
    suit_x = 3
    suit_y = rank_y + font_size + 1
    draw.text((suit_x, suit_y), suit_text, fill=color, font=suit_font)

    # Draw border
    draw.rectangle([0, 0, width - 1, height - 1], outline=(180, 180, 180), width=1)

    # Convert to OpenCV BGR format
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_corner_template(rank: str, suit: str, width: int = 25, height: int = 35,
                              style: str = "default") -> np.ndarray:
    """Generate just the corner portion of a card (rank + suit indicator).
    This is often more reliable for template matching since it's the most
    distinctive part of the card."""

    suits = POKERSTARS_SUITS if style == "pokerstars" else SUITS

    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    suit_info = suits[suit]
    color = suit_info['color']
    rank_text = RANK_DISPLAY[rank]
    suit_text = suit_info['symbol']

    font_size = max(10, width // 2)
    suit_font_size = max(8, width // 3)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        suit_font = ImageFont.truetype("arial.ttf", suit_font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            suit_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", suit_font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
            suit_font = ImageFont.load_default()

    draw.text((2, 1), rank_text, fill=color, font=font)
    draw.text((2, font_size + 2), suit_text, fill=color, font=suit_font)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_all_templates(output_dir: str, width: int = 50, height: int = 70,
                            style: str = "default", corners_only: bool = False):
    """Generate template images for all 52 cards.

    Args:
        output_dir: Directory to save templates
        width: Card width
        height: Card height
        style: Visual style
        corners_only: If True, generate only corner templates
    """
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for rank in RANKS:
        for suit_key in SUITS:
            card_name = f"{rank}{suit_key}"

            if corners_only:
                img = generate_corner_template(rank, suit_key, width, height, style)
            else:
                img = generate_card_image(rank, suit_key, width, height, style)

            filepath = os.path.join(output_dir, f"{card_name}.png")
            # Use imencode+open for Unicode path support (å/ä/ö)
            success, buf = cv2.imencode('.png', img)
            if success:
                with open(filepath, 'wb') as f:
                    f.write(buf.tobytes())
            count += 1

    print(f"Generated {count} card templates in {output_dir}")
    print(f"  Style: {style}, Size: {width}x{height}, Corners only: {corners_only}")


def capture_templates_from_screen(output_dir: str):
    """Interactive tool to capture card templates directly from a poker client.

    Opens a screen capture window where you can click on each card to
    create a template from the actual poker client graphics.
    """
    import mss

    os.makedirs(output_dir, exist_ok=True)
    sct = mss.mss()

    print("=== CARD TEMPLATE CAPTURE ===")
    print("This tool captures card images directly from your poker client.")
    print("1. Open your poker client and go to a table")
    print("2. A screenshot will be taken")
    print("3. For each card, click top-left then bottom-right of the card")
    print("4. Then type the card name (e.g. 'As' for Ace of spades)")
    print()

    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    remaining_cards = []
    for rank in RANKS:
        for suit_key in SUITS:
            card_name = f"{rank}{suit_key}"
            if not os.path.exists(os.path.join(output_dir, f"{card_name}.png")):
                remaining_cards.append(card_name)

    if not remaining_cards:
        print("All 52 card templates already exist!")
        return

    print(f"{len(remaining_cards)} cards remaining to capture.")
    print("Press 'S' to skip a card, 'Q' to quit.")

    display = frame.copy()
    # Scale down for display if too large
    h, w = display.shape[:2]
    if w > 1920:
        scale = 1920 / w
        display = cv2.resize(display, None, fx=scale, fy=scale)

    points = []

    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
            cv2.imshow("Capture Cards", display)

    cv2.imshow("Capture Cards", display)
    cv2.setMouseCallback("Capture Cards", click_handler)

    for card_name in remaining_cards:
        print(f"\nCapture card: {card_name}")
        print("  Click top-left corner, then bottom-right corner of the card")
        points.clear()

        while len(points) < 2:
            key = cv2.waitKey(100)
            if key == ord('s') or key == ord('S'):
                print(f"  Skipped {card_name}")
                break
            if key == ord('q') or key == ord('Q'):
                print("Quitting capture.")
                cv2.destroyAllWindows()
                sct.close()
                return

        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            # Account for display scaling
            h_orig, w_orig = frame.shape[:2]
            h_disp, w_disp = display.shape[:2]
            sx = w_orig / w_disp
            sy = h_orig / h_disp

            x1_orig = int(x1 * sx)
            y1_orig = int(y1 * sy)
            x2_orig = int(x2 * sx)
            y2_orig = int(y2 * sy)

            card_img = frame[
                min(y1_orig, y2_orig):max(y1_orig, y2_orig),
                min(x1_orig, x2_orig):max(x1_orig, x2_orig),
            ]

            filepath = os.path.join(output_dir, f"{card_name}.png")
            cv2.imwrite(filepath, card_img)
            print(f"  Saved {card_name} ({card_img.shape[1]}x{card_img.shape[0]})")

            # Draw rectangle on display
            cv2.rectangle(display, (min(x1, x2), min(y1, y2)),
                         (max(x1, x2), max(y1, y2)), (0, 255, 0), 1)
            cv2.imshow("Capture Cards", display)

    cv2.destroyAllWindows()
    sct.close()

    # Count how many we have
    count = sum(
        1 for r in RANKS for s in SUITS
        if os.path.exists(os.path.join(output_dir, f"{r}{s}.png"))
    )
    print(f"\nDone! {count}/52 card templates saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate card templates for poker OCR")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for templates")
    parser.add_argument("--size", type=str, default="50x70",
                       help="Template size WxH (default: 50x70)")
    parser.add_argument("--style", type=str, choices=["default", "pokerstars"],
                       default="pokerstars", help="Visual style")
    parser.add_argument("--corners", action="store_true",
                       help="Generate corner-only templates (more reliable)")
    parser.add_argument("--capture", action="store_true",
                       help="Capture templates from screen instead of generating")
    parser.add_argument("--both", action="store_true",
                       help="Generate both full and corner templates")
    args = parser.parse_args()

    if not HAS_DEPS:
        print("Error: opencv-python and Pillow are required")
        print("Install with: pip install opencv-python Pillow")
        sys.exit(1)

    # Default output directory
    if args.output is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output = os.path.join(base_dir, "models", "card_templates")

    if args.capture:
        capture_templates_from_screen(args.output)
        return

    w, h = map(int, args.size.split("x"))

    if args.both:
        # Full cards
        generate_all_templates(args.output, w, h, args.style, corners_only=False)
        # Corner templates in subdirectory
        corners_dir = os.path.join(args.output, "corners")
        generate_all_templates(corners_dir, w // 2, h // 2, args.style, corners_only=True)
    else:
        generate_all_templates(args.output, w, h, args.style, corners_only=args.corners)

    print("\nTemplates ready! You can now run the poker assistant.")
    print("For better accuracy, also capture templates from your poker client:")
    print(f"  python {__file__} --capture")


if __name__ == "__main__":
    main()
