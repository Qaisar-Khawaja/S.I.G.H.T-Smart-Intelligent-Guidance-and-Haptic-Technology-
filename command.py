def generate_command(action):
    """
    Maps a decision action to the single-character code sent to the Pico.

    C = close hazard  -> fast / high-intensity pulse
    M = medium hazard -> slower / medium pulse
    S = safe          -> motor off
    """
    commands = {
        "STOP": "C",
        "WARN_LEFT": "L",
        "WARN_RIGHT": "R",
        "SAFE": "S",
    }
    return commands.get(action, "S")