"""
Temporal stabilisation for cane commands.

THE PROBLEM
    YOLO's per-frame output is noisy in ways the physical world is not.
    A stationary person drops out of detection for a frame or two when
    confidence dips; a bounding box jitters a few pixels either side of
    a distance threshold; an object straddling the frame-thirds boundary
    flips between LEFT and CENTER. Acting on the raw per-frame command
    therefore produces rapid state churn -- C, S, C, L, C -- and every
    one of those transitions restarts a clip and resets the vibration.
    The result is a cane that chatters instead of informing.

THE FIX
    A proposed command must persist for several consecutive frames
    before it is committed. Until then it is only a candidate, and the
    cane keeps doing what it was doing.

    The two directions are deliberately asymmetric:

        escalating   (something got MORE dangerous)  -> commit fast
        de-escalating (something got LESS dangerous) -> commit slowly

    The costs are not symmetric. A late warning means walking into an
    obstacle; a late all-clear means a moment of unnecessary caution.
    So the system is quick to warn and slow to reassure.

    Escalation is tallied by URGENCY, not by exact letter. If frames
    alternate L, C, L, C while the cane is in S, every one of them
    agrees that things got worse, so they accumulate together and the
    worst of them commits. Tallying by exact letter would let the two
    candidates reset each other forever and the user would be warned
    about nothing at all.

    De-escalation and lateral moves (L <-> R) still require the SAME
    command to persist, since there is no urgency to accumulate and a
    person crossing the field of view shouldn't produce direction
    chatter as they pass through centre.

NOTE ON UNITS
    Thresholds are counted in FRAMES, so the wall-clock delay scales
    with frame rate. At 15fps, 8 frames is about half a second; if YOLO
    slows to 5fps that becomes 1.6s and may feel sluggish. Retune
    FRAMES_TO_RELAX if your measured frame rate changes a lot.
"""

import time


# Ordering that defines what "more urgent" means.
URGENCY = {"S": 0, "L": 1, "R": 1, "C": 2}

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
        # Tally by urgency, not by letter, so alternating proposals
        # (L, C, L, C) still accumulate toward a commit.
        if raw_urgency > stable_urgency:
            self._escalate_count += 1
            self._relax_count = 0
            self._relax_candidate = None

            # Remember the worst thing seen during this escalation.
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

        # ── Case 3: less urgent, or a lateral move (L <-> R) ──
        # Require the SAME command to persist for the slow threshold.
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