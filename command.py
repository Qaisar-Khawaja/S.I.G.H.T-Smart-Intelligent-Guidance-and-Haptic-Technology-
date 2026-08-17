"""
Translates an action label into the code sent over the wire.

WIRE FORMAT
    Two characters: urgency then direction.

        char 1   urgency    M = medium, C = close
        char 2   direction  L = left, C = centre, R = right

        S        safe -- the one single-character code, since there is
                 no direction to report when there is no hazard

    Full set:  S, ML, MC, MR, CL, CC, CR

    The Pico reads character one to choose a vibration pattern and
    character two to choose a clip. It never has to know what the
    combination means -- two independent lookups, no interpretation.

    Positional encoding is why "C" appearing in both columns is not
    ambiguous: CC is close-centre, MC is medium-centre.

    Keep this in sync with the Pico firmware. A code emitted here that
    the Pico doesn't recognise is dropped silently, which looks like
    dead hardware rather than a mismatch.
"""

URGENCY_CODE = {
    "CLOSE":  "C",
    "MEDIUM": "M",
}

DIRECTION_CODE = {
    "LEFT":   "L",
    "CENTER": "C",
    "RIGHT":  "R",
}

SAFE_CODE = "S"


def generate_command(action):
    """
    "CLOSE_LEFT" -> "CL",  "MEDIUM_CENTER" -> "MC",  "SAFE" -> "S".

    Anything unrecognised falls back to SAFE. Failing quiet is the right
    default here: a malformed code should leave the cane silent rather
    than buzzing about a hazard we can't describe.
    """
    if action == "SAFE":
        return SAFE_CODE

    urgency, _, direction = action.partition("_")

    urgency_char = URGENCY_CODE.get(urgency)
    direction_char = DIRECTION_CODE.get(direction)

    if urgency_char is None or direction_char is None:
        return SAFE_CODE

    return urgency_char + direction_char