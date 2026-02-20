"""
Push/Fold charts for short-stack tournament play.

Based on ICM-adjusted Nash equilibrium ranges.
When your stack is < 15 big blinds, poker simplifies to
push (all-in) or fold decisions.
"""


# Push ranges by position and stack size (in big blinds)
# True = push, these are the MINIMUM hands to push with
# Format: set of hand notations

# 10-15 BB push ranges
PUSH_15BB = {
    "UTG": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s",
        "KQs", "KJs",
        "AKo", "AQo",
    },
    "HJ": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s",
        "KQs", "KJs", "KTs", "K9s",
        "QJs", "QTs",
        "JTs",
        "AKo", "AQo", "AJo", "ATo",
        "KQo",
    },
    "CO": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s",
        "QJs", "QTs", "Q9s",
        "JTs", "J9s",
        "T9s",
        "98s",
        "AKo", "AQo", "AJo", "ATo", "A9o",
        "KQo", "KJo",
        "QJo",
    },
    "BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s",
        "JTs", "J9s", "J8s",
        "T9s", "T8s",
        "98s", "97s",
        "87s", "86s",
        "76s", "75s",
        "65s",
        "54s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o",
        "KQo", "KJo", "KTo",
        "QJo", "QTo",
        "JTo",
    },
    "SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s",
        "KQs", "KJs", "KTs", "K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s",
        "QJs", "QTs", "Q9s", "Q8s", "Q7s", "Q6s",
        "JTs", "J9s", "J8s", "J7s",
        "T9s", "T8s", "T7s",
        "98s", "97s", "96s",
        "87s", "86s", "85s",
        "76s", "75s",
        "65s", "64s",
        "54s", "53s",
        "43s",
        "AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o", "A5o",
        "KQo", "KJo", "KTo", "K9o",
        "QJo", "QTo", "Q9o",
        "JTo", "J9o",
        "T9o",
    },
}

# 6-10 BB push ranges (wider)
PUSH_10BB = {
    "UTG": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s", "A4s",
        "KQs", "KJs", "KTs", "K9s",
        "QJs", "QTs",
        "JTs",
        "T9s",
        "98s",
        "AKo", "AQo", "AJo", "ATo",
        "KQo",
    },
    "HJ": PUSH_15BB["CO"].copy(),
    "CO": PUSH_15BB["BTN"].copy(),
    "BTN": PUSH_15BB["SB"].copy(),
    "SB": PUSH_15BB["SB"] | {
        # Even wider from SB with 6-10BB
        "K2s", "Q5s", "Q4s", "J6s", "T6s", "95s", "84s", "74s", "63s",
        "A4o", "A3o", "A2o", "K8o", "K7o", "Q8o", "J8o",
    },
}

# 3-6 BB push ranges (very wide — almost any two cards from late position)
PUSH_6BB = {
    "UTG": PUSH_10BB["HJ"].copy(),
    "HJ": PUSH_10BB["CO"].copy(),
    "CO": PUSH_10BB["SB"].copy(),
    "BTN": PUSH_10BB["SB"] | {
        "K2s", "Q5s", "Q4s", "Q3s", "J6s", "J5s", "T6s", "95s",
        "84s", "74s", "63s", "52s", "42s",
        "A4o", "A3o", "A2o", "K8o", "K7o", "K6o",
        "Q8o", "Q7o", "J8o", "J7o", "T8o",
    },
    "SB": {
        # Push almost everything heads-up with <6BB
        f"{r1}{r2}{s}" for r1 in "AKQJT98765432"
        for r2 in "AKQJT98765432"
        for s in ("s", "o", "")
        if r1 != r2 or s == ""
    },
}

# Call ranges vs all-in (by position)
CALL_ALLIN = {
    "BB_vs_SB": {
        "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66",
        "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s",
        "KQs", "KJs", "KTs",
        "QJs", "QTs",
        "JTs",
        "AKo", "AQo", "AJo", "ATo",
        "KQo",
    },
    "BB_vs_BTN": {
        "AA", "KK", "QQ", "JJ", "TT", "99",
        "AKs", "AQs", "AJs", "ATs",
        "KQs",
        "AKo", "AQo",
    },
    "BB_vs_CO": {
        "AA", "KK", "QQ", "JJ", "TT",
        "AKs", "AQs", "AJs",
        "AKo", "AQo",
    },
    "BB_vs_UTG": {
        "AA", "KK", "QQ", "JJ",
        "AKs", "AQs",
        "AKo",
    },
}


# BB push range: tighter than SB since BB already has money invested
# and only shoves when checked to (not opening)
PUSH_15BB["BB"] = PUSH_15BB["CO"].copy()  # Tighter — already invested 1BB
PUSH_10BB["BB"] = PUSH_10BB["CO"].copy()
PUSH_6BB["BB"] = PUSH_6BB["CO"].copy()

# Position aliases for normalization
_POSITION_ALIASES = {
    "UTG+1": "UTG", "UTG+2": "UTG", "MP": "UTG", "MP+1": "UTG",
    "BUTTON": "BTN", "DEALER": "BTN", "D": "BTN",
    "SMALL": "SB", "BIG": "BB",
    "HIJACK": "HJ", "CUTOFF": "CO",
}


def _normalize_position(pos: str) -> str:
    """Normalize position string to standard format."""
    pos = pos.upper().strip()
    return _POSITION_ALIASES.get(pos, pos)


def get_push_range(position: str, stack_bb: float) -> set:
    """Get the push range for a given position and stack size.

    Args:
        position: Player position (UTG, HJ, CO, BTN, SB)
        stack_bb: Stack size in big blinds

    Returns:
        Set of hand notations to push with
    """
    pos = _normalize_position(position)

    if stack_bb <= 6:
        chart = PUSH_6BB
    elif stack_bb <= 10:
        chart = PUSH_10BB
    else:
        chart = PUSH_15BB

    return chart.get(pos, chart.get("UTG", set()))


def should_push(hand_notation: str, position: str, stack_bb: float) -> bool:
    """Check if a hand should be pushed all-in.

    Args:
        hand_notation: Hand notation like 'AKs', 'JJ', 'T9o'
        position: Player position
        stack_bb: Stack size in big blinds

    Returns:
        True if the hand is in the push range
    """
    push_range = get_push_range(position, stack_bb)
    return hand_notation in push_range


def get_call_range(your_position: str, pusher_position: str) -> set:
    """Get the call range vs an all-in from a specific position.

    Args:
        your_position: Your position (usually BB)
        pusher_position: Position of the player who went all-in

    Returns:
        Set of hand notations to call with
    """
    your_pos = _normalize_position(your_position)
    pusher_pos = _normalize_position(pusher_position)
    key = f"{your_pos}_vs_{pusher_pos}"
    return CALL_ALLIN.get(key, CALL_ALLIN.get("BB_vs_UTG", set()))


def should_call_allin(hand_notation: str, your_position: str,
                      pusher_position: str) -> bool:
    """Check if a hand should call an all-in.

    Args:
        hand_notation: Your hand notation
        your_position: Your position
        pusher_position: All-in raiser's position

    Returns:
        True if you should call
    """
    call_range = get_call_range(your_position, pusher_position)
    return hand_notation in call_range
