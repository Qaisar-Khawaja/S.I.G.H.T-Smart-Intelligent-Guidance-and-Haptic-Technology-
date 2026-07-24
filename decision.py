def cane_decision(object_name, direction, distance):

    dangerous_objects = [
        "person",
        "car",
        "bicycle",
        "chair",
        "dog"
    ]


    if object_name in dangerous_objects:

        if distance == "CLOSE":

            if direction == "LEFT":
                return "WARN_LEFT"

            elif direction == "RIGHT":
                return "WARN_RIGHT"

            elif direction == "CENTER":
                return "STOP"


    return "SAFE"