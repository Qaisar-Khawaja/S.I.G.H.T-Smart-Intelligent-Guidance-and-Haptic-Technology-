"""
Describes what each real cane-sweep video (data/videos/videoN.mov) is
meant to test, for use by eval_dataset_b.py (to organize/label results
per-video) and the report (to pick qualitative before/after examples).

This is scene/intent metadata, not measured data -- durations/fps are
confirmed via ffprobe; the rest is what the video was staged to show.
"""

VIDEOS = {
    "video1": {
        "duration_s": 21.0,
        "scene": "chairs",
        "tests": "Baseline near-field obstacle detection (chairs).",
        "synthetic_lighting_variants": True,
    },
    "video2": {
        "duration_s": 29.0,
        "scene": "chairs",
        "tests": "Large object suddenly in the path, short clip -- tests "
                 "detection speed/robustness for a big, close obstacle.",
        "synthetic_lighting_variants": True,
    },
    "video3": {
        "duration_s": 12.4,
        "scene": "small objects on ground + people standing",
        "tests": "Small trip hazards on the ground and standing people. "
                 "Motivates urgency scaling by object size/distance in "
                 "decision.py (not a restoration concern): small objects "
                 "leave the frame as the user gets close, so they need a "
                 "more urgent haptic/audio signal earlier than large "
                 "objects that stay in view longer.",
    },
    "video4": {
        "duration_s": 29.2,
        "scene": "small objects on ground + people standing",
        "tests": "Same scenario as video3 (longer take).",
    },
    "video5": {
        "duration_s": 17.5,
        "scene": "chairs, slightly darker environment",
        "tests": "Short/tall large objects (chairs) under dim lighting -- "
                 "real low-light degradation.",
    },
    "video6": {
        "duration_s": 23.7,
        "scene": "walking people, slightly darker environment",
        "tests": "Moving objects (people walking) under dim lighting -- "
                 "real motion blur + low light combined.",
    },
    "video7": {
        "duration_s": 11.5,
        "scene": "table edge",
        "tests": "Hero example: a table edge is invisible to a physical "
                 "cane tip (hollow underneath), but the camera can flag "
                 "it from a distance before the user makes contact.",
    },
}
