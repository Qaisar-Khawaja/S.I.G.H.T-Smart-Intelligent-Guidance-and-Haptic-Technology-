"""
Temporal stabilisation for cane commands.

THE PROBLEM
    YOLO's per-frame output is noisy in ways the physical world is not.
    A stationary person drops out of detection for a frame or two when
    confidence dips; a bounding box jitters a few pixels either side of
    a distance threshold; an object straddling the frame-thirds boundary
    flips between LEFT and CENTER. Acting on the raw per-frame command
    therefore produces rapid state churn, and every transition restarts
    a clip and resets the vibration. The result is a cane that chatters
    instead of informing.

THE FIX
    A proposed command must persist for several consecutive frames
    before it is committed. Until then it is only a candidate, and the
    cane keeps doing what it was doing.

    The two directions are deliberately asymmetric:

        escalating    (something got MORE dangerous)  -> commit fast
        de-escalating (something got LESS dangerous)  -> commit slowly

    The costs are not symmetric. A late warning means walking into an
    obstacle; a late all-clear means a moment of unnecessary caution.
    So the system is quick to warn and slow to reassure.

    Escalation is tallied by URGENCY, not by exact code. If frames
    alternate ML, CR while the cane is in S, both agree that things got
    worse, so they accumulate together and the worse of them commits.
    Tallying by exact code would let the two candidates reset each
    other forever and the user would be warned about nothing at all.

TWO-DIMENSIONAL CODES
    Codes are urgency + direction (see command.py). Only the urgency
    half participates in the escalate/relax decision -- a change of
    direction at the same urgency (ML -> MR, or MC -> ML) is neither an
    escalation nor a de-escalation, so it routes through the slow path.

    That is deliberate and it is what stabilises direction: a person
    drifting across the frame-thirds boundary produces LEFT/CENTER
    flicker, and requiring the new direction to persist stops the
    speaker announcing a new side every few frames.

ONE STREAM, BOTH CHANNELS
    The Pico drives speech and vibration from a single state variable
    fed by this stabilised stream, so the two channels cannot drift out
    of step. Stabilising the command stabilises both.

NOTE ON UNITS
    Thresholds are counted in FRAMES, so the wall-clock delay scales
    with frame rate. At 15fps, 8 frames is about half a second; if YOLO
    slows to 5fps that becomes 1.6s and may feel sluggish. Retune
    FRAMES_TO_RELAX if your measured frame rate changes a lot.
"""

import time


# Ordering that defines what "more urgent" means. All three MEDIUM codes
# sit at tier 1 and all three CLOSE codes at tier 2: direction does not
# affect urgency, which is the whole point of separating the dimensions.
URGENCY = {
    "S":  0,
    "ML": 1, "MC": 1, "MR": 1,
    "CL": 2, "CC": 2, "CR": 2,
}

FRAMES_TO_ESCALATE = 2      # ~0.13s at 15fps
FRAMES_TO_RELAX    = 8      # ~0.53s at 15fps

HEARTBEAT_SECONDS  = 4.0    # resend the committed command this often even
                            # when unchanged, so a dropped UDP packet can't
                            # strand the cane buzzing or silent


class CommandStabilizer:
    """
    Feed it the raw command for each frame; it tells you what to
    actually transmit.

        stabilizer = CommandStabilizer()

        to_send = stabilizer.update(raw_command)
        if to_send:
            send_command(to_send)

    update() returns the command string when something should go out on
    the wire, or None when it shouldn't. It handles three jobs that were
    previously scattered through main.py: hysteresis, de-duplication,
    and the heartbeat resend.

    Read stabilizer.state for the currently committed command (useful
    for on-screen display) and stabilizer.just_changed to tell a real
    state change apart from a routine heartbeat.
    """

    def __init__(self,
                 escalate_frames=FRAMES_TO_ESCALATE,
                 relax_frames=FRAMES_TO_RELAX,
                 heartbeat=HEARTBEAT_SECONDS,
                 initial="S"):

        self.escalate_needed = escalate_frames
        self.relax_needed    = relax_frames
        self.heartbeat       = heartbeat

        self.state        = initial     # committed command
        self.just_changed = False       # True only on the frame it commits

        self._escalate_count   = 0
        self._relax_count      = 0
        self._pending          = None   # worst command seen while escalating
        self._relax_candidate  = None   # command being proposed while relaxing

        self._last_sent    = None
        self._last_sent_at = 0.0

    # ── public ─────────────────────────────────────────────

    def update(self, raw_command, now=None):
        """
        Advance one frame. Returns a command to transmit, or None.
        """
        if now is None:
            now = time.time()

        self.just_changed = False
        self._advance(raw_command)

        # A genuine state change always goes out immediately.
        if self.state != self._last_sent:
            self.just_changed = True
            self._last_sent = self.state
            self._last_sent_at = now
            return self.state

        # Otherwise only the periodic heartbeat.
        if now - self._last_sent_at >= self.heartbeat:
            self._last_sent_at = now
            return self.state

        return None

    def reset(self, state="S"):
        """Force the committed state, clearing all pending evidence."""
        self.state = state
        self._escalate_count = 0
        self._relax_count = 0
        self._pending = None
        self._relax_candidate = None

    # ── internal ───────────────────────────────────────────

    def _advance(self, raw):
        """The hysteresis state machine. Updates self.state in place."""

        raw_urgency    = URGENCY.get(raw, 0)
        stable_urgency = URGENCY.get(self.state, 0)

        # ── Case 1: more urgent than the committed state ──
        # Tally by urgency, not by exact code, so alternating proposals
        # (ML, CR, ML, CR) still accumulate toward a commit.
        if raw_urgency > stable_urgency:
            self._escalate_count += 1
            self._relax_count = 0
            self._relax_candidate = None

            # Remember the worst thing seen during this escalation. The
            # direction that commits is the one attached to that worst
            # reading, which is the hazard actually being escaped.
            if (self._pending is None
                    or raw_urgency >= URGENCY.get(self._pending, 0)):
                self._pending = raw

            if self._escalate_count >= self.escalate_needed:
                self.state = self._pending
                self._escalate_count = 0
                self._pending = None

        # ── Case 2: agrees with the committed state ──
        # Any agreeing frame wipes out accumulated contrary evidence.
        # This is what makes brief detection dropouts harmless.
        elif raw == self.state:
            self._escalate_count = 0
            self._relax_count = 0
            self._pending = None
            self._relax_candidate = None

        # ── Case 3: less urgent, OR the same urgency with a different
        # direction (ML -> MR). Both require the SAME code to persist
        # for the slow threshold, which is what stops the speaker
        # flipping sides as an object drifts across a boundary.
        else:
            self._escalate_count = 0
            self._pending = None

            if raw == self._relax_candidate:
                self._relax_count += 1
            else:
                self._relax_candidate = raw
                self._relax_count = 1

            if self._relax_count >= self.relax_needed:
                self.state = raw
                self._relax_count = 0
                self._relax_candidate = None