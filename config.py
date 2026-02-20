"""
Central configuration for the Poker AI Assistant.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # === CAPTURE ===
    capture_mode: str = "screen"  # "screen" or "camera"
    camera_index: int = 0
    capture_fps: float = 2.0  # Frames per second to analyze
    screen_region: dict = None  # {top, left, width, height} for screen capture

    # === VISION ===
    template_dir: str = "models/card_templates"
    regions_file: str = "models/table_regions.json"
    ocr_confidence_threshold: float = 0.75
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # === STRATEGY ===
    equity_simulations: int = 10000  # Monte Carlo iterations
    exploit_min_hands: int = 20  # Min hands before exploiting opponent

    # === LLM ===
    llm_model: str = "poker-ai"  # Ollama model name (custom poker model)
    llm_temperature: float = 0.3
    llm_enabled: bool = True

    # === UI ===
    ui_mode: str = "window"  # "window" or "console"

    # === DATABASE ===
    opponent_db_path: str = "profiles/opponents.db"

    # === GAME ===
    hero_seat: int = 0            # Hero's seat index in player_regions
    default_sb: float = 0.5       # Default small blind
    default_bb: float = 1.0       # Default big blind
    max_stack: float = 1_000_000  # Max valid stack (for sanity check)

    # === ANTI-DETECTION ===
    min_response_delay: float = 2.0  # Minimum seconds before showing advice
    max_response_delay: float = 8.0  # Maximum seconds (randomized)
    simulate_human_errors: bool = False  # Occasionally show suboptimal advice
    human_error_rate: float = 0.05  # 5% of the time
