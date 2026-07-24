def generate_command(action):

    commands = {

        "WARN_LEFT": "L",

        "WARN_RIGHT": "R",

        "STOP": "C",

        "SAFE": "S"

    }

    return commands.get(action, "S")