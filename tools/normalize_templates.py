"""Normalize all card templates to a consistent size.

Problem: Templates captured manually have wildly different sizes (73x101 to
254x351 pixels). This causes inconsistent template matching because the
corner cache gets resized from very different source resolutions.

Solution: Find the actual card body in each template, crop it, and resize
all to a standard size. This ensures consistent corner-cache quality.

Usage:
    python tools/normalize_templates.py
    python tools/normalize_templates.py --target-width 70 --target-height 100
    python tools/normalize_templates.py --dry-run
"""

import cv2
import numpy as np
import os
import sys
import shutil
import argparse

# Standard card aspect ratio is ~0.7 (width/height)
DEFAULT_WIDTH = 70
DEFAULT_HEIGHT = 100

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "card_templates_live")
BACKUP_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "card_templates_live_backup")


def load_image(path):
    """Load image using cv2.imdecode to handle Swedish path chars."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path, img):
    """Save image using cv2.imencode to handle Swedish path chars."""
    success, buf = cv2.imencode('.png', img)
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def find_card_body(img):
    """Find the actual card rectangle within the image.

    Cards have a white/light body that stands out against any background.
    Uses contour detection to find the largest bright region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Try multiple thresholds to find the card body
    for thresh_val in [140, 120, 160, 100]:
        _, bright = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Card body should be a significant portion of the image
        if area > h * w * 0.15:
            x, y, cw, ch = cv2.boundingRect(largest)
            # Ensure reasonable aspect ratio for a card (0.5 to 0.9)
            aspect = cw / max(ch, 1)
            if 0.4 < aspect < 1.0:
                return x, y, cw, ch

    # Fallback: use the full image
    return 0, 0, w, h


def normalize_template(img, target_w, target_h):
    """Crop to card body and resize to target dimensions."""
    x, y, cw, ch = find_card_body(img)

    # Add small padding (2px) to not clip the edge
    h, w = img.shape[:2]
    x = max(0, x - 2)
    y = max(0, y - 2)
    cw = min(w - x, cw + 4)
    ch = min(h - y, ch + 4)

    cropped = img[y:y+ch, x:x+cw]

    # Resize to standard dimensions
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def main():
    parser = argparse.ArgumentParser(description='Normalize card templates to consistent size')
    parser.add_argument('--target-width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--target-height', type=int, default=DEFAULT_HEIGHT)
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without saving')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup')
    args = parser.parse_args()

    template_dir = os.path.abspath(TEMPLATE_DIR)
    backup_dir = os.path.abspath(BACKUP_DIR)

    if not os.path.isdir(template_dir):
        print(f"Template directory not found: {template_dir}")
        sys.exit(1)

    # List all PNG files
    files = sorted([f for f in os.listdir(template_dir) if f.endswith('.png')])
    print(f"Found {len(files)} templates in {template_dir}")
    print(f"Target size: {args.target_width}x{args.target_height}")

    if not files:
        print("No templates found!")
        sys.exit(1)

    # Backup
    if not args.dry_run and not args.no_backup:
        if os.path.exists(backup_dir):
            print(f"Backup already exists at {backup_dir} — skipping backup")
        else:
            print(f"Creating backup at {backup_dir}")
            shutil.copytree(template_dir, backup_dir)
            print(f"Backup complete ({len(files)} files)")

    # Process each template
    success = 0
    failed = []
    for fname in files:
        path = os.path.join(template_dir, fname)
        img = load_image(path)
        if img is None:
            print(f"  FAILED to load: {fname}")
            failed.append(fname)
            continue

        orig_h, orig_w = img.shape[:2]
        normalized = normalize_template(img, args.target_width, args.target_height)

        if args.dry_run:
            print(f"  {fname:8s}  {orig_w:4d}x{orig_h:<4d} -> {args.target_width}x{args.target_height}")
        else:
            if save_image(path, normalized):
                success += 1
            else:
                print(f"  FAILED to save: {fname}")
                failed.append(fname)

    if args.dry_run:
        print(f"\nDry run complete. Would normalize {len(files)} templates.")
    else:
        print(f"\nNormalized {success}/{len(files)} templates to {args.target_width}x{args.target_height}")

    if failed:
        print(f"Failed: {', '.join(failed)}")

    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
