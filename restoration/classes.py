"""
Shared object-class definitions for image restoration evaluation.

This is the same filtered COCO subset used to curate data/clean/ --
obstacle-relevant classes a cane user would actually encounter. Kept
here as the single source of truth so eval_dataset_a.py and any future
scripts agree on which classes count as "relevant" instead of each
redefining the set.
"""

RELEVANT_CLASSES = {0, 1, 2, 3, 5, 7, 9, 10, 11, 13, 16, 24, 26, 28, 39, 56, 57, 58, 60, 67}

CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 9: "traffic light", 10: "fire hydrant", 11: "stop sign",
    13: "bench", 16: "dog", 24: "backpack", 26: "handbag", 28: "suitcase",
    39: "bottle", 56: "chair", 57: "couch", 58: "potted plant",
    60: "dining table", 67: "cell phone",
}
