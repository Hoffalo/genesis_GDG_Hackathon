import random


class MyBot():
    def __init__(self):
        self.name = "MyBot"

    # IMPLEMENT YOUR METHODS HERE

    # Modify but don't rename this method
    def act(self, info):
        # receives info dictionary from the game (for the proper player)
        # at the end of this method you should return a dictionary of moves, for example:
        actions = {
            "forward": True,
            "right": False,
            "down": False,
            "left": False,
            "rotate": 0,
            "shoot": True
            } # in this example will make the bot go forward and shoot
        # always include even non used/changed variables

        "--- example random act function ---"
        direction = random.choice(["forward", "right", "down", "left"])
        actions = {
            "forward": direction == "forward",
            "right": direction == "right",
            "down": direction == "down",
            "left": direction == "left",
            "rotate": random.randint(-180, 180),  # Rotate randomly
            "shoot": random.choice([True, False])  # Randomly decide to shoot or not
        }

        return actions