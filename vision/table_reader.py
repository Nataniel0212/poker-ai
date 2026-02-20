"""
Vision module — reads the poker table from a screenshot/frame.
Uses OpenCV template matching for cards and Tesseract OCR for text.

This module is designed to be configurable per poker site/skin by
adjusting the ROI (Region of Interest) definitions and card templates.
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import pytesseract
except ImportError:
    pytesseract = None


@dataclass
class TableRegions:
    """Defines pixel regions for each element on the poker table.
    All coordinates are (x, y, width, height) relative to the table image.

    You must configure these for your specific poker client/skin.
    Use the calibration tool (vision/calibrate.py) to set these up.
    """
    # Hero's hole cards (two card positions)
    hero_card_1: tuple = (0, 0, 0, 0)
    hero_card_2: tuple = (0, 0, 0, 0)

    # Community cards (5 positions)
    community_cards: list = field(default_factory=lambda: [(0, 0, 0, 0)] * 5)

    # Pot display
    pot_text: tuple = (0, 0, 0, 0)

    # Player areas — list of dicts with sub-regions
    # Each player: {name, stack, bet, seat_index}
    player_regions: list = field(default_factory=list)

    # Dealer button
    dealer_button_search_area: tuple = (0, 0, 0, 0)

    # Per-seat dealer button search regions
    dealer_button_regions: list = field(default_factory=list)

    # Action buttons (fold, check, call, raise, bet)
    action_buttons_area: tuple = (0, 0, 0, 0)

    # Bet slider
    bet_slider_area: tuple = (0, 0, 0, 0)


@dataclass
class TableReading:
    """The result of reading a poker table frame."""
    hero_cards: list = field(default_factory=list)      # ['As', 'Kh']
    community_cards: list = field(default_factory=list)  # ['Js', 'Ts', '3d']
    pot: float = 0.0
    players: list = field(default_factory=list)          # [{name, stack, bet, seat}]
    dealer_seat: int = -1
    available_actions: list = field(default_factory=list) # ['fold', 'call', 'raise']
    confidence: float = 0.0  # Overall confidence in the reading


# Card rank and suit mappings
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']  # spades, hearts, diamonds, clubs


class TableReader:
    """Reads poker table state from screen captures."""

    def __init__(self, template_dir: str = None, regions: TableRegions = None,
                 tesseract_cmd: str = None):
        """
        Args:
            template_dir: Path to directory with card template images
            regions: TableRegions defining where elements are on screen
            tesseract_cmd: Path to tesseract executable
        """
        self.regions = regions or TableRegions()
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), "..", "models", "card_templates"
        )
        self.card_templates = {}
        self._max_stack = 1_000_000  # Reject stacks above this (OCR noise)
        self._frame_count = 0
        self._cached_players = []
        self._cached_dealer = -1
        self._player_update_interval = 5  # Read players every Nth frame
        if tesseract_cmd and pytesseract:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._load_templates()

    # Standard template size — all templates normalized to this on load.
    # Chosen to be larger than typical community card ROI (~75x105) so we
    # always scale DOWN (INTER_AREA) which preserves detail better than up.
    TEMPLATE_SIZE = (150, 210)  # (width, height)

    def _load_templates(self):
        """Load card template images from disk.

        Tries live-captured templates first (models/card_templates_live/),
        then falls back to synthetic templates (models/card_templates/).
        All templates are normalized to TEMPLATE_SIZE on load for consistent
        matching regardless of original capture resolution.
        """
        # Directories to search (priority order)
        # PS-captured templates (auto-saved from live game) take highest priority
        base = os.path.dirname(self.template_dir)
        ps_dir = os.path.join(base, "card_templates_ps")
        live_dir = os.path.join(base, "card_templates_live")
        dirs = [ps_dir, live_dir, self.template_dir]

        tw, th = self.TEMPLATE_SIZE

        for rank in RANKS:
            for suit in SUITS:
                card_name = f"{rank}{suit}"
                if card_name in self.card_templates:
                    continue  # Already loaded from higher-priority dir
                for tdir in dirs:
                    if not os.path.exists(tdir):
                        continue
                    for ext in ('.png', '.jpg', '.bmp'):
                        path = os.path.join(tdir, f"{card_name}{ext}")
                        if os.path.exists(path):
                            with open(path, 'rb') as f:
                                data = np.frombuffer(f.read(), np.uint8)
                            template = cv2.imdecode(data, cv2.IMREAD_COLOR)
                            if template is not None:
                                # Normalize to standard size
                                template = cv2.resize(template, (tw, th),
                                                      interpolation=cv2.INTER_AREA)
                                self.card_templates[card_name] = template
                            break
                    if card_name in self.card_templates:
                        break

        # Pre-build corner cache (top-left 45% of each normalized template)
        self._corner_cache = {}
        for card_name, template in self.card_templates.items():
            h, w = template.shape[:2]
            corner = template[0:int(h * 0.45), 0:int(w * 0.45)]
            self._corner_cache[card_name] = corner

        if self.card_templates:
            live_count = sum(1 for c in self.card_templates
                           if os.path.exists(os.path.join(live_dir, f"{c}.png")))
            print(f"Loaded {len(self.card_templates)} card templates ({live_count} live, normalized to {tw}x{th})")
        else:
            print("No card templates found — card detection will rely on OCR only")

    def read_table(self, frame: np.ndarray) -> TableReading:
        """Read all elements from a poker table frame.

        Args:
            frame: BGR numpy array of the poker table

        Returns:
            TableReading with all detected elements
        """
        reading = TableReading()
        self._frame_count += 1

        # Read hero's cards (fast — template matching only)
        reading.hero_cards = self._read_hero_cards(frame)

        # Read community cards (fast — template matching only)
        reading.community_cards = self._read_community_cards(frame)

        # Read pot (moderate — single OCR call)
        reading.pot = self._read_pot(frame)

        # Read player info + dealer (slow — many OCR calls)
        # Only update every Nth frame to maintain good FPS
        if self._frame_count % self._player_update_interval == 1:
            self._cached_players = self._read_players(frame)
            self._cached_dealer = self._find_dealer(frame)
        reading.players = self._cached_players
        reading.dealer_seat = self._cached_dealer

        # Read available actions
        reading.available_actions = self._read_actions(frame)

        # Calculate overall confidence
        conf_scores = []
        if reading.hero_cards:
            conf_scores.append(1.0)
        if reading.pot > 0:
            conf_scores.append(1.0)
        if reading.dealer_seat >= 0:
            conf_scores.append(1.0)
        reading.confidence = sum(conf_scores) / max(len(conf_scores), 1)

        return reading

    def _detect_card(self, roi: np.ndarray) -> Optional[str]:
        """Detect a card using edge-based template matching with sliding window.

        Uses Canny edge detection on both template and ROI before matching.
        This makes matching invariant to background color (works for both
        white-background community cards and dark-background hero cards).

        Args:
            roi: Cropped image of a card area

        Returns:
            Card string like 'As' or None if not detected
        """
        if roi is None or roi.size == 0 or not self._corner_cache:
            return None

        if not self._is_card_present(roi):
            return None

        rh, rw = roi.shape[:2]
        if rh < 20 or rw < 15:
            return None

        # Search area: top-left 65% of ROI (card corner must be in this area)
        search_h = int(rh * 0.65)
        search_w = int(rw * 0.65)
        search_area = roi[0:search_h, 0:search_w]

        # Template corner target: ~35% of ROI (smaller than search area = sliding)
        target_h = max(int(rh * 0.35), 15)
        target_w = max(int(rw * 0.35), 10)

        if target_h >= search_h or target_w >= search_w:
            return None

        # Convert search area to edge map (invariant to background color)
        search_gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        search_edges = cv2.Canny(search_gray, 50, 150)

        best_match = None
        best_score = 0
        threshold = 0.35  # Lower threshold for edge matching (edges are sparse)

        for card_name, corner_tmpl in self._corner_cache.items():
            tmpl = cv2.resize(corner_tmpl, (target_w, target_h),
                              interpolation=cv2.INTER_AREA)

            # Convert template to edge map too
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            tmpl_edges = cv2.Canny(tmpl_gray, 50, 150)

            result = cv2.matchTemplate(search_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score and max_val > threshold:
                best_score = max_val
                best_match = card_name

        return best_match

    def _is_card_present(self, roi: np.ndarray) -> bool:
        """Check if a card is actually present (not just green felt or dark background).

        A card has HIGH CONTRAST: bright rank/suit text on a background (either
        white card or dark PS panel). Green felt has LOW contrast (uniform color).
        """
        if roi is None or roi.size == 0:
            return False

        h, w = roi.shape[:2]
        if h < 5 or w < 5:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Green felt detection
        green = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 220]))
        green_ratio = cv2.countNonZero(green) / max(h * w, 1)
        # If >55% green, it's just felt (no card)
        if green_ratio > 0.55:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Very dark AND low contrast = empty dark area (no card)
        if gray.mean() < 25 and gray.std() < 15:
            return False

        # A card should have SOME contrast (text/symbols on background)
        # Pure green felt or dark panel has stddev < 20
        # A card with rank text has stddev > 25
        if gray.std() < 12:
            return False

        # Check for bright pixels (rank text is usually bright: >150)
        bright_ratio = np.sum(gray > 150) / max(h * w, 1)
        # Check for colored pixels (suit colors on 4-color deck)
        sat_channel = hsv[:, :, 1]
        colored_ratio = np.sum(sat_channel > 60) / max(h * w, 1)

        # Card has either bright white text OR colored text
        if bright_ratio > 0.02 or colored_ratio > 0.03:
            return True

        # Still might be a card if there's moderate contrast
        if gray.std() > 25:
            return True

        return False

    def _detect_card_ocr(self, roi: np.ndarray) -> Optional[str]:
        """Detect a card using OCR on the top-left corner (rank + suit).

        Works with any poker client style since it reads the actual text
        rather than matching templates. Upscales small corners for better OCR.
        """
        if roi is None or roi.size == 0 or pytesseract is None:
            return None

        h, w = roi.shape[:2]
        if h < 10 or w < 10:
            return None

        # Check if there's actually a card here
        if not self._is_card_present(roi):
            return None

        # Crop top-left corner where rank is shown (~40% width, ~40% height)
        rank_corner = roi[1:int(h * 0.40), 1:int(w * 0.40)]
        if rank_corner.size == 0:
            return None

        # Upscale 4x for better OCR accuracy on small text
        scale = 4
        rank_big = cv2.resize(rank_corner, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(rank_big, cv2.COLOR_BGR2GRAY)

        # Try multiple thresholds and PSM modes — white text on dark card
        rank = None
        psm_modes = ['--psm 10', '--psm 8']  # single char, then single word
        for thresh_val in [140, 120, 160, 100, 180]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            # Also try inverted (dark text on light background)
            for img in [thresh, cv2.bitwise_not(thresh)]:
                for psm in psm_modes:
                    text = pytesseract.image_to_string(
                        img,
                        config=f'{psm} -c tessedit_char_whitelist=23456789TJQKA10'
                    ).strip()
                    if not text:
                        continue
                    text = text.upper().replace(' ', '')
                    if text.startswith('10'):
                        rank = 'T'
                        break
                    ch = text[0]
                    if ch in '23456789TJQKA':
                        rank = ch
                        break
                    # Common OCR misreads
                    ocr_fix = {'O': 'T', 'I': 'J', 'L': 'J', 'B': '8',
                               'S': '5', 'G': '6', 'D': '0', 'Z': '2',
                               'P': '9', 'R': 'K'}
                    if ch in ocr_fix:
                        fixed = ocr_fix[ch]
                        if fixed == '0' or fixed == 'T':
                            rank = 'T'
                        elif fixed in '23456789':
                            rank = fixed
                        elif fixed in 'JKA':
                            rank = fixed
                        break
                if rank:
                    break
            if rank:
                break

        if not rank:
            return None

        suit = self._detect_suit(roi)
        return f"{rank}{suit}"

    def _detect_suit(self, roi: np.ndarray) -> str:
        """Detect card suit using color + shape analysis (standard 2-color deck).

        Standard deck:
          RED suits: Hearts ♥ and Diamonds ♦ — distinguished by shape
          BLACK suits: Spades ♠ and Clubs ♣ — distinguished by shape

        Step 1: Detect if red or black using the suit symbol area
        Step 2: Use shape analysis to distinguish within each color group
        """
        h, w = roi.shape[:2]

        # Crop ONLY the suit symbol, BELOW the rank text.
        # On PokerStars cards the rank character occupies ~0-20% height,
        # the suit symbol is at ~22-42% height, left ~35% width.
        # We start at 25% to definitively clear the rank text.
        suit_y1 = int(h * 0.25)
        suit_y2 = int(h * 0.45)
        suit_x2 = int(w * 0.38)
        suit_area = roi[suit_y1:suit_y2, 0:suit_x2]
        if suit_area.size == 0 or suit_area.shape[0] < 3 or suit_area.shape[1] < 3:
            # Fallback: use wider area
            suit_area = roi[int(h * 0.18):int(h * 0.45), 0:int(w * 0.40)]
        if suit_area.size == 0:
            suit_area = roi

        hsv = cv2.cvtColor(suit_area, cv2.COLOR_BGR2HSV)
        total = suit_area.shape[0] * suit_area.shape[1]

        # Detect red pixels (hearts or diamonds)
        # Use permissive thresholds (sat>=50, val>=40) to catch dark hero card suits
        red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([168, 50, 40]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(red1) + cv2.countNonZero(red2)
        red_ratio = red_pixels / max(total, 1)

        # Red suit detected → distinguish hearts vs diamonds by shape
        if red_ratio > 0.02:
            return self._distinguish_heart_diamond(suit_area, hsv)

        # Black/no color → distinguish spades vs clubs by shape
        return self._distinguish_spade_club(suit_area)

    def _distinguish_heart_diamond(self, suit_area: np.ndarray,
                                    hsv: np.ndarray) -> str:
        """Distinguish hearts from diamonds using shape analysis (2-color deck).

        Heart (♥): rounded top with concavity (two bumps), wider shape
        Diamond (♦): pointed at top and bottom, narrow/angular, high solidity

        Key insight: we select the best suit-shaped contour (compact, right size)
        rather than just the largest, to avoid rank text contamination.
        """
        red1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([168, 50, 40]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        # Morphological close to merge nearby pixels into solid shapes
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Upscale with INTER_LINEAR for smoother edges (not NEAREST which is blocky)
        red_big = cv2.resize(red_mask, None, fx=4, fy=4,
                             interpolation=cv2.INTER_LINEAR)
        # Re-threshold after linear interpolation (creates gray pixels)
        _, red_big = cv2.threshold(red_big, 127, 255, cv2.THRESH_BINARY)
        # Smooth edges with Gaussian blur + re-threshold
        red_big = cv2.GaussianBlur(red_big, (3, 3), 0)
        _, red_big = cv2.threshold(red_big, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(red_big, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 'h'

        # Filter contours: suit symbol should be reasonably sized and compact
        img_h, img_w = red_big.shape[:2]
        img_area = img_h * img_w
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 20:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            area_ratio = area / max(img_area, 1)
            if area_ratio < 0.03 or area_ratio > 0.85:
                continue
            aspect = cw / max(ch, 1)
            if aspect < 0.3 or aspect > 2.5:
                continue
            fill = area / max(cw * ch, 1)
            candidates.append((c, area, fill, aspect, y))

        if not candidates:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 20:
                return 'h'
            candidates = [(largest, cv2.contourArea(largest), 0.5, 1.0, 0)]

        # Pick the best candidate: prefer lower position (more likely suit, not rank)
        candidates.sort(key=lambda c: (-c[4], -c[1]))
        best = candidates[0][0]

        hull = cv2.convexHull(best)
        hull_area = cv2.contourArea(hull)
        cont_area = cv2.contourArea(best)
        solidity = cont_area / max(hull_area, 1)

        x, y, cw, ch = cv2.boundingRect(best)
        aspect = cw / max(ch, 1)

        # Method 1: Vertex-based detection (most reliable)
        # Diamond (♦) approximates to exactly 4 vertices (quadrilateral)
        # Heart (♥) has 5+ vertices due to the concavity at top
        epsilon = 0.04 * cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, epsilon, True)

        # Strong diamond signal: exactly 4 vertices + reasonable solidity
        # Diamond must also not be much wider than tall (aspect < 1.5)
        if len(approx) == 4 and solidity > 0.70 and aspect < 1.5:
            return 'd'

        # Strong heart signal: 5+ vertices (concavity adds extra vertex)
        if len(approx) >= 6:
            return 'h'

        # Method 2: Top profile analysis (PRIMARY method for ambiguous cases)
        # Heart: dip in center-top (between two bumps) → center < sides
        # Diamond: peak in center-top (pointed) → center > sides
        top_strip = red_big[y:y + max(ch // 4, 1), x:x + cw]
        if top_strip.size > 0:
            col_sums = np.sum(top_strip > 0, axis=0)
            if len(col_sums) > 4:
                mid = len(col_sums) // 2
                quarter = max(len(col_sums) // 4, 1)
                center_mass = np.mean(col_sums[mid - quarter:mid + quarter])
                side_mass = (np.mean(col_sums[:quarter]) + np.mean(col_sums[-quarter:])) / 2
                if side_mass > 0:
                    ratio = center_mass / side_mass
                    if ratio < 0.7:
                        return 'h'  # Center dip = heart
                    if ratio > 1.3:
                        return 'd'  # Center peak = diamond

        # Method 3: Width analysis
        # Heart (♥) is wider than tall (the two bumps spread it out)
        # Diamond (♦) is roughly equal or taller than wide
        # At 5 vertices: if wider than tall → heart, else diamond
        if len(approx) == 5:
            if aspect > 1.2:
                return 'h'
            if aspect < 0.85:
                return 'd'

        # Method 4: Convexity defects — heart has significant concavity at top
        hull_indices = cv2.convexHull(best, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(best, hull_indices)
            if defects is not None:
                sig_defects = sum(1 for d in defects if d[0][3] > 150)
                if sig_defects >= 1:
                    return 'h'

        # Default for 5 vertices: likely heart (diamond rarely gets 5)
        if len(approx) >= 5:
            return 'h'

        # Final fallback
        return 'd' if aspect < 1.0 else 'h'

    def _distinguish_spade_club(self, suit_area: np.ndarray) -> str:
        """Distinguish spades from clubs using shape analysis.

        Spade (♠): pointed top, rounded bottom, stem at base → high solidity
        Club (♣): three round lobes → lower solidity, wider aspect ratio
        """
        gray = cv2.cvtColor(suit_area, cv2.COLOR_BGR2GRAY)

        # Try multiple thresholds for dark pixel detection
        for thresh_val in [80, 100, 120]:
            dark = cv2.inRange(gray, 0, thresh_val)
            dark_ratio = cv2.countNonZero(dark) / max(suit_area.shape[0] * suit_area.shape[1], 1)
            if dark_ratio > 0.05:
                break

        dark_big = cv2.resize(dark, None, fx=4, fy=4,
                              interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(dark_big, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 's'  # Default to spade when no shape found

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        cont_area = cv2.contourArea(largest)
        solidity = cont_area / max(hull_area, 1)

        x, y, cw, ch = cv2.boundingRect(largest)
        aspect = cw / max(ch, 1)

        # Spade: taller than wide (aspect < 1.0), high solidity
        # Club: wider due to 3 lobes (aspect >= 1.0), low solidity
        if solidity > 0.65:
            return 's'
        if aspect > 1.2 and solidity < 0.55:
            return 'c'

        # Fallback: check convexity defects
        hull_indices = cv2.convexHull(largest, returnPoints=False)
        if hull_indices is not None and len(hull_indices) > 3:
            defects = cv2.convexityDefects(largest, hull_indices)
            if defects is not None:
                sig_defects = sum(1 for d in defects if d[0][3] > 500)
                if sig_defects >= 2:
                    return 'c'

        return 's'

    def _crop_region(self, frame: np.ndarray, region: tuple) -> np.ndarray:
        """Crop a region from the frame."""
        x, y, w, h = region
        if w == 0 or h == 0:
            return np.array([])
        return frame[y:y+h, x:x+w]

    def _auto_capture_template(self, roi: np.ndarray, card_name: str):
        """Auto-save a card ROI as a PS-specific template for future matching.

        Only saves if we don't already have a PS-captured template for this card.
        This builds up accurate templates over time from actual live game frames.
        """
        if roi is None or roi.size == 0 or not card_name:
            return
        if len(card_name) != 2:
            return

        # Save to a PS-specific template directory
        ps_dir = os.path.join(os.path.dirname(self.template_dir), "card_templates_ps")
        os.makedirs(ps_dir, exist_ok=True)

        path = os.path.join(ps_dir, f"{card_name}.png")
        if os.path.exists(path):
            return  # Already have this card

        # Save the ROI as template
        ok, buf = cv2.imencode('.png', roi)
        if ok:
            with open(path, 'wb') as f:
                f.write(buf)

            # Also update the in-memory template cache
            tw, th = self.TEMPLATE_SIZE
            template = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)
            self.card_templates[card_name] = template
            h, w = template.shape[:2]
            self._corner_cache[card_name] = template[0:int(h * 0.45), 0:int(w * 0.45)]

    def _read_hero_cards(self, frame: np.ndarray) -> list:
        """Read hero's two hole cards. Uses OCR first (more reliable), template as fallback."""
        cards = []
        for region in [self.regions.hero_card_1, self.regions.hero_card_2]:
            roi = self._crop_region(frame, region)
            card = self._detect_card_ocr(roi)
            if card:
                self._auto_capture_template(roi, card)
            else:
                card = self._detect_card(roi)
            if card:
                cards.append(card)
        return cards

    def _read_community_cards(self, frame: np.ndarray) -> list:
        """Read community cards (flop, turn, river). Uses OCR first, template as fallback."""
        cards = []
        for region in self.regions.community_cards:
            roi = self._crop_region(frame, region)
            card = self._detect_card_ocr(roi)
            if card:
                self._auto_capture_template(roi, card)
            else:
                card = self._detect_card(roi)
            if card:
                cards.append(card)
        return cards

    def _read_text(self, frame: np.ndarray, region: tuple,
                   mode: str = "amount") -> str:
        """Read text from a region using Tesseract OCR.

        Args:
            frame: Full table frame
            region: (x, y, w, h) region to read
            mode: 'amount' for numbers/money, 'name' for player names
        """
        if pytesseract is None:
            return ""

        roi = self._crop_region(frame, region)
        if roi.size == 0:
            return ""

        # Preprocess for OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Fixed threshold works better than OTSU on green felt
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        # Scale up for better OCR
        scaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        if mode == "name":
            config = "--psm 7"
        else:
            config = "--psm 7 -c tessedit_char_whitelist=0123456789.$,kKmMBb: "

        text = pytesseract.image_to_string(scaled, config=config).strip()

        return text

    def _parse_amount(self, text: str) -> float:
        """Parse a money amount from OCR text like '$12.50', '12,500', or 'Pott: 14 665'."""
        if not text:
            return 0.0

        # Remove common prefixes (PokerStars Swedish: "Pott:", English: "Pot:")
        import re
        text = re.sub(r'^[A-Za-z]+:\s*', '', text)

        # Clean up common OCR artifacts
        text = text.replace(",", "").replace(" ", "")
        text = text.replace("$", "").replace("€", "")
        text = text.lower()

        # Handle k/m suffixes
        multiplier = 1
        if text.endswith("k"):
            multiplier = 1000
            text = text[:-1]
        elif text.endswith("m"):
            multiplier = 1000000
            text = text[:-1]

        try:
            return float(text) * multiplier
        except ValueError:
            return 0.0

    def _read_pot(self, frame: np.ndarray) -> float:
        """Read the pot amount."""
        # Use name mode since pot text may contain "Pott:" prefix
        text = self._read_text(frame, self.regions.pot_text, mode="name")
        return self._parse_amount(text)

    def _clean_name(self, name: str) -> str:
        """Clean OCR artifacts from player name.

        Common artifacts:
        - Dealer button "D" read as leading "v", "d", "D"
        - Bet/action text bleeding: "bd", "Bd", prefix fragments
        - Trailing action words: "Raise", "Call", "Fold"
        """
        import re
        if not name:
            return ""

        # Remove non-alphanumeric chars from start/end
        name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
        name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
        name = name.strip()

        # Remove common OCR prefix artifacts (1-2 char noise before real name)
        # "v Nataniel02" → "Nataniel02", "bd Raise" → "Raise" (then filtered)
        # "D PlayerName" → "PlayerName"
        name = re.sub(r'^[vVdDbB][dD]?\s+', '', name)

        # Remove trailing action words that bleed into name region
        name = re.sub(r'\s+(Fold|Call|Raise|Check|Bet|All.?[Ii]n|Sit\s*Out).*$',
                      '', name, flags=re.IGNORECASE)

        # Remove leading single char followed by space (likely dealer button noise)
        name = re.sub(r'^[a-zA-Z]\s+', '', name)

        return name.strip()

    def _is_valid_name(self, name: str) -> bool:
        """Check if OCR text is a valid player name (not green felt noise)."""
        if not name or len(name) <= 2:
            return False
        lower = name.lower().strip()
        # Common noise patterns from green felt
        noise = {"re", "ae", "er", "ar", "or", "en", "et", "po", "bd", "vd"}
        if lower in noise:
            return False
        # Poker action words (not player names)
        actions = {"fold", "call", "raise", "check", "bet", "allin", "all-in",
                   "sit out", "sitting out", "away", "dealer"}
        if lower in actions:
            return False
        # Too short after cleaning is suspicious
        if len(name) <= 1:
            return False
        return True

    def _read_players(self, frame: np.ndarray) -> list:
        """Read all player info (name, stack, current bet)."""
        players = []
        for i, pregion in enumerate(self.regions.player_regions):
            player = {
                "seat": i,
                "name": "",
                "stack": 0.0,
                "bet": 0.0,
            }

            if "name" in pregion:
                name_text = self._read_text(frame, pregion["name"], mode="name")
                name_text = self._clean_name(name_text)
                if self._is_valid_name(name_text):
                    player["name"] = name_text

            if "stack" in pregion:
                stack_text = self._read_text(frame, pregion["stack"])
                stack = self._parse_amount(stack_text)
                # Sanity check: reject absurd values (OCR noise)
                if stack > self._max_stack:
                    stack = 0.0
                player["stack"] = stack

            if "bet" in pregion:
                bet_text = self._read_text(frame, pregion["bet"])
                player["bet"] = self._parse_amount(bet_text)

            if player["name"] or player["stack"] > 0:
                players.append(player)

        return players

    def _find_dealer(self, frame: np.ndarray) -> int:
        """Find the dealer position by detecting who posted blinds.

        Reads bet amounts for each player. The small blind is the smallest
        non-zero bet, and the dealer sits one position before the SB
        (in 6-max: seat before SB going clockwise).
        """
        # Read bets for all players
        bets = []
        for i, pregion in enumerate(self.regions.player_regions):
            if "bet" not in pregion:
                bets.append((i, 0.0))
                continue
            bet_text = self._read_text(frame, pregion["bet"])
            bet = self._parse_amount(bet_text)
            bets.append((i, bet))

        # Find seats with non-zero bets (potential blinds)
        active_bets = [(seat, bet) for seat, bet in bets if bet > 0]

        if len(active_bets) < 2:
            return -1

        # Sort by bet size — smallest is SB, second is BB
        active_bets.sort(key=lambda x: x[1])
        sb_seat = active_bets[0][0]

        # Dealer is one seat before SB (going backwards in seat order)
        num_seats = len(self.regions.player_regions)
        dealer_seat = (sb_seat - 1) % num_seats

        return dealer_seat

    def _detect_dealer_button(self, roi: np.ndarray) -> float:
        """Score how likely a region contains the dealer button.

        Returns a confidence score (0-1). Looks for:
        1. A bright white/cream circular blob
        2. Roughly circular shape (aspect ratio ~1.0)
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Dealer button is bright white/cream: low saturation, high value
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([30, 60, 255]))

        # Clean up noise
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Dealer button should be a reasonable size (not tiny noise, not huge)
        roi_area = roi.shape[0] * roi.shape[1]
        area_ratio = area / max(roi_area, 1)
        if area_ratio < 0.02 or area_ratio > 0.5:
            return 0.0

        # Check circularity: perimeter^2 / (4 * pi * area) ≈ 1.0 for circle
        perimeter = cv2.arcLength(largest, True)
        if perimeter == 0:
            return 0.0
        circularity = (4 * 3.14159 * area) / (perimeter * perimeter)

        # Check aspect ratio of bounding rect (~1.0 for circle)
        x, y, w, h = cv2.boundingRect(largest)
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        # Combined score: good circle = high circularity + aspect near 1.0
        if circularity < 0.5 or aspect < 0.6:
            return 0.0

        # Score based on circularity and area ratio
        return circularity * area_ratio * 10  # Scale up for comparison

    def _read_actions(self, frame: np.ndarray) -> list:
        """Read available action buttons."""
        # This would detect which buttons are visible (fold, check, call, raise, bet)
        # For now, returns common defaults
        return ["fold", "check", "call", "raise"]


class TableCalibrator:
    """Interactive tool to configure TableRegions for a specific poker client."""

    def __init__(self):
        self.regions = TableRegions()
        self.current_frame = None
        self.click_points = []

    def calibrate_from_frame(self, frame: np.ndarray) -> TableRegions:
        """Interactive calibration — user clicks to define regions.

        Args:
            frame: A screenshot of the poker table

        Returns:
            Configured TableRegions
        """
        self.current_frame = frame.copy()
        display = frame.copy()

        instructions = [
            ("Hero Card 1", "hero_card_1"),
            ("Hero Card 2", "hero_card_2"),
            ("Community Card 1", "community_0"),
            ("Community Card 2", "community_1"),
            ("Community Card 3", "community_2"),
            ("Community Card 4", "community_3"),
            ("Community Card 5", "community_4"),
            ("Pot Text", "pot_text"),
        ]

        print("=== TABLE CALIBRATION ===")
        print("For each element, click the TOP-LEFT corner, then BOTTOM-RIGHT corner.")
        print("Press 'S' to skip, 'R' to reset clicks, 'Q' to quit.\n")

        try:
            for label, field_name in instructions:
                print(f"Click to define: {label}")
                points = self._get_two_clicks(display, label)

                if points is None:
                    print(f"  Skipped {label}")
                    continue

                (x1, y1), (x2, y2) = points
                region = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

                if field_name.startswith("community_"):
                    idx = int(field_name.split("_")[1])
                    self.regions.community_cards[idx] = region
                else:
                    setattr(self.regions, field_name, region)

                # Draw rectangle on display
                cv2.rectangle(display, (region[0], region[1]),
                             (region[0] + region[2], region[1] + region[3]),
                             (0, 255, 0), 2)
                cv2.putText(display, label, (region[0], region[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                print(f"  {label}: {region}")
        finally:
            cv2.destroyAllWindows()

        return self.regions

    def _get_two_clicks(self, frame, label, timeout_seconds=120):
        """Get two click points from the user.

        Args:
            frame: Image to display
            label: Current element being calibrated
            timeout_seconds: Max wait time before auto-skip (default 120s)
        """
        import time
        points = []
        start_time = time.time()

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("Calibration", frame)

        cv2.imshow("Calibration", frame)
        cv2.setMouseCallback("Calibration", click_handler)

        while len(points) < 2:
            key = cv2.waitKey(100)
            if key == ord('s') or key == ord('S'):
                return None
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return None
            if key == ord('r') or key == ord('R'):
                # Reset points to start over
                points.clear()
                print(f"  Reset — click again for {label}")
            if time.time() - start_time > timeout_seconds:
                print(f"  Timeout after {timeout_seconds}s — skipping {label}")
                return None

        return points

    def save_regions(self, filepath: str):
        """Save configured regions to a JSON file."""
        import json
        data = {
            "hero_card_1": self.regions.hero_card_1,
            "hero_card_2": self.regions.hero_card_2,
            "community_cards": self.regions.community_cards,
            "pot_text": self.regions.pot_text,
            "player_regions": self.regions.player_regions,
            "dealer_button_search_area": self.regions.dealer_button_search_area,
            "dealer_button_regions": self.regions.dealer_button_regions,
            "action_buttons_area": self.regions.action_buttons_area,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Regions saved to {filepath}")

    def load_regions(self, filepath: str) -> TableRegions:
        """Load configured regions from a JSON file."""
        import json
        with open(filepath) as f:
            data = json.load(f)

        self.regions.hero_card_1 = tuple(data.get("hero_card_1", (0, 0, 0, 0)))
        self.regions.hero_card_2 = tuple(data.get("hero_card_2", (0, 0, 0, 0)))
        self.regions.community_cards = [tuple(c) for c in data.get("community_cards", [(0,0,0,0)]*5)]
        self.regions.pot_text = tuple(data.get("pot_text", (0, 0, 0, 0)))
        self.regions.player_regions = data.get("player_regions", [])
        self.regions.dealer_button_search_area = tuple(data.get("dealer_button_search_area", (0,0,0,0)))
        self.regions.dealer_button_regions = [tuple(r) for r in data.get("dealer_button_regions", [])]
        self.regions.action_buttons_area = tuple(data.get("action_buttons_area", (0,0,0,0)))

        return self.regions
