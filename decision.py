"""
Hazard decision logic for the smart cane.

Turns a single detected object (name, direction, distance) into a
numeric "how urgent is this" score, and an action label.

TWO DIMENSIONS, KEPT SEPARATE
    Urgency  = how close the hazard is       -> drives the vibration
    Direction = which way the hazard lies    -> drives the speech

    These are independent facts and the action label now carries both:

        SAFE
        MEDIUM_LEFT    MEDIUM_CENTER    MEDIUM_RIGHT
        CLOSE_LEFT     CLOSE_CENTER     CLOSE_RIGHT

    The previous scheme collapsed them, so a MEDIUM obstacle dead ahead
    and a CLOSE obstacle to the left both produced "STOP" -- the cane
    said the same thing for two quite different situations. Splitting
    them means the vibration answers "how bad?" and the voice answers
    "where?", and neither has to encode the other.

SCORING IS SEPARATE FROM OUTPUT
    hazard_score() still exists and still uses direction as a small
    tiebreaker, but that is only about ARBITRATION -- which of several
    objects in a frame is the one worth reacting to. main.py scores
    every detection, keeps the highest, and only then asks for an
    action label. A centred obstacle outranks a side one at equal
    distance because it's the one you're walking into.
"""

import important_objects

# Base urgency by distance tier. FAR (or anything unrecognised) = 0 = ignore.
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
    Priority score for one detected object. 0 means "ignore it".
    Used only to pick the winning object within a frame.
    """
    if object_name not in important_objects.important_objects:
        return 0

    base = DISTANCE_SCORE.get(distance, 0)
    if base == 0:
        return 0

    return base + DIRECTION_BONUS.get(direction, 0)


def cane_decision(object_name, direction, distance):
    """
    Action label for the winning object: URGENCY_DIRECTION, or SAFE.

    Note that direction passes straight through rather than being
    folded into the urgency tier. A CLOSE hazard on the left stays
    CLOSE_LEFT -- the user is told both that it's urgent and which way
    to move, which is more actionable than a bare instruction to stop.
    """
    score = hazard_score(object_name, direction, distance)

    if score == 0:
        return "SAFE"

    urgency = "CLOSE" if score >= DISTANCE_SCORE["CLOSE"] else "MEDIUM"

    return f"{urgency}_{direction}"