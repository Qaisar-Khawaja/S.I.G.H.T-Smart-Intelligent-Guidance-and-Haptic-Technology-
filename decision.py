"""
Hazard decision logic for the smart cane.

Turns a single detected object (name, direction, distance) into a
numeric "how urgent is this" score, and an action label.

TWO DIMENSIONS
    Urgency  = how close the hazard is       -> drives the vibration
    Direction = which way the hazard lies    -> drives the speech

        SAFE
        MEDIUM_LEFT    MEDIUM_CENTER    MEDIUM_RIGHT
        CLOSE_LEFT     CLOSE_CENTER     CLOSE_RIGHT


SCORING IS SEPARATE FROM OUTPUT
    hazard_score() uses direction as a small
    tiebreaker, but that is only about ARBITRATION -- which of several
    objects in a frame is the one worth reacting to. main.py scores
    every detection, keeps the highest, and only then asks for an
    action label. A centred obstacle outranks a side one at equal
    distance.
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
    to move.
    """
    score = hazard_score(object_name, direction, distance)

    if score == 0:
        return "SAFE"

    urgency = "CLOSE" if score >= DISTANCE_SCORE["CLOSE"] else "MEDIUM"

    return f"{urgency}_{direction}"