"""
Hazard decision logic for the smart cane.
Turns a single detected object (name, direction, distance) into a
numeric "how urgent is this" score, and a simple action label.

Design:
- Distance sets the base urgency tier (CLOSE > MEDIUM > FAR/ignored).
  This alone decides how intense the haptic buzz is.
- Direction only matters as a *tiebreaker* between hazards that are at
  the SAME distance tier: CENTER gets a small bonus because that's the
  direction the person is actually walking into. It never outweighs
  distance (the bonus is tiny compared to the gap between tiers).
- main.py scores every object detected in a frame and picks the single
  highest score to act on, so two simultaneous hazards don't fight
  each other over the serial line.
"""

DANGEROUS_OBJECTS = [
    "bottle",
    "cell phone",
]

# Base urgency by distance tier. FAR (or anything unrecognized) = 0 = ignore.
DISTANCE_SCORE = {
    "CLOSE": 100,
    "MEDIUM": 50,
    "FAR": 0,
}

# Small tiebreaker bonus so that, at the same distance, an object dead
# ahead outranks one off to the side. Kept small on purpose so it can
# never push a MEDIUM hazard above a CLOSE one.
DIRECTION_BONUS = {
    "CENTER": 2,
    "LEFT": 0,
    "RIGHT": 0,
}


def hazard_score(object_name, direction, distance):
    """
    Returns a priority score for one detected object.
    0 means "not a hazard, ignore it."
    """
    if object_name not in DANGEROUS_OBJECTS:
        return 0

    base = DISTANCE_SCORE.get(distance, 0)
    if base == 0:
        return 0

    return base + DIRECTION_BONUS.get(direction, 0)


def cane_decision(object_name, direction, distance):
    """
    Convenience wrapper: score -> action label.
    """
    score = hazard_score(object_name, direction, distance)

    if score >= DISTANCE_SCORE["CLOSE"]:
        return "STOP"
    elif score >= DISTANCE_SCORE["MEDIUM"]:
        return "WARN"
    else:
        return "SAFE"
